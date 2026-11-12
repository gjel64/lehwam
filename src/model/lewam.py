import torch
import torch.nn as nn
import torch.optim as optim # for gradient based search
import types
import sys # for error message

from .encoder import Encoder
from .predictor import MoT
from .modules import SIGReg

# ------------------------------------------------ LeWAM ------------------------------------------------


class leWAM(nn.Module):
    def __init__(self, cfg, act_dim, device):
        super().__init__()
        self.encoder = Encoder(cfg.emb_dim, act_dim)
        self.MoT = MoT(cfg.emb_dim, act_dim, cfg.n_head, 4, cfg.n_block)

        self.next_state_query = nn.Parameter(torch.randn(1, 1, cfg.emb_dim) * 0.02)

        self.act_dim = act_dim

        # not beautiful xd
        mask = torch.ones(19, 19, device=device) * float('-inf')
        mask[:, :2] = 0
        mask[2, 2] = 0
        mask[2, 4:9] = 0
        mask[3, 3] = 0
        mask[4, 4] = 0
        mask[5, 4:6] = 0
        mask[6, 4:7] = 0
        mask[7, 4:8] = 0
        mask[8, 4:9] = 0
        mask[9:, 3] = 0
        mask[9, 9] = 0
        mask[10, 9:11] = 0
        mask[11, 9:12] = 0
        mask[12, 9:13] = 0
        mask[13, 9:14] = 0
        mask[14, 9:15] = 0
        mask[15, 9:16] = 0
        mask[16, 9:17] = 0
        mask[17, 9:18] = 0
        mask[18, 9:19] = 0
        self.register_buffer("mask", mask)

        # Planning parameters
        self.k = cfg.planning.k
        self.n_euler_step = cfg.planning.n_euler_step
        self.n_rollout_step = cfg.planning.n_rollout_step
        self.n_gd_step = cfg.planning.n_gd_step
        self.lr_for_gd = cfg.planning.lr_for_gd
        self.lambda_prox = cfg.planning.lambda_prox
        self.a_max = cfg.planning.a_max

        # SIGReg
        self.sigreg = SIGReg()
        self.sigreg_coef = cfg.sigreg_coef

        # MoT adaptator
        self.hist_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.goal_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.hist_pos = nn.Parameter(torch.zeros(1, 2, cfg.emb_dim))     # zeros as LeWAM
        self.action_pos = nn.Parameter(torch.zeros(1, 10, cfg.emb_dim))
        self.state_type = nn.Parameter(torch.randn(3, cfg.emb_dim))      # 0 history, 1 next state, 2 goal
        self.action_type = nn.Parameter(torch.randn(2, cfg.emb_dim))     # 0 clean, 1 noisy

# ------------------------------- Usefull methods -------------------------------

    def _attn_mask(self, history_pad):
        """(19, 19) -> (B, 1, 19, 19)"""
        mask = self.mask.expand(history_pad.shape[0], 1, 19, 19).clone()
        mask[history_pad[:, 0], :, :, 0] = float("-inf")
        return mask

    def predictor_adaptor(self, z_history, z_goal, z_next, z_action, type="full"):
        """ 
        adapt the zs to MoT by adding position, proj (z->usable_z), type
        takes the 'raw' embeddings of an encoder and adapt it to predictor
        -> very usefull w/ a pretrained encoder  
        (B, 2, z_dim), (B, 1, z_dim), (B, 1, z_dim), (B, 15, z_dim) -> (B, 19, z_dim) tokens for the MoT (on full)
        """
        # the type is used because at generate, it calc actions everytimes but images once
        match type:
            case "full":
                h = self.hist_proj(z_history) + self.hist_pos + self.state_type[0]     # (B, 2, z_dim)
                n = z_next + self.state_type[1]                                        # (B, 1, z_dim)
                g = self.goal_proj(z_goal) + self.state_type[2]                        # (B, 1, z_dim)
                clean = z_action[:, :5] + self.action_pos[:, :5] + self.action_type[0] # (B, 5, z_dim)
                noisy = z_action[:, 5:] + self.action_pos + self.action_type[1]        # (B, 10, z_dim)
                return torch.cat([h, n, g, clean, noisy], dim=1)                       # (B, 19, z_dim)
            case "actions":
                clean = z_action[:, :5] + self.action_pos[:, :5] + self.action_type[0]
                noisy = z_action[:, 5:] + self.action_pos + self.action_type[1]
                return torch.cat([clean, noisy], dim=1)
            case "images":
                h = self.hist_proj(z_history) + self.hist_pos + self.state_type[0]
                n = z_next + self.state_type[1]
                g = self.goal_proj(z_goal) + self.state_type[2]
                return torch.cat([h, n, g], dim=1)

# ------------------------------- Forward -------------------------------

    def forward(self, o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad):
        """
        o_history : (B, 2, 3, 224, 224)
        o_target :  (B, 1, 3, 224, 224)
        o_goal :    (B, 1, 3, 224, 224)
    
        actions :   (B, 10, act_dim)
        """
        losses = {
            "SIGReg" : [],
            "MSE" : []
        }
        B = o_history.shape[0]

        # Create the flow matching action, a mixture of noise and the true action with tau as the mixing coefficient
        # first 5 actions are the true actions, last 10 are the flow matching actions
        tau = torch.rand(B, device=o_history.device)
        t = tau[:, None, None]
        eps = torch.randn_like(actions, device=o_history.device) 
        x_tau = (1 - t) * eps + t * actions # noise and true part ((1-tau) and tau)
        a = torch.cat([actions[:, :5], x_tau], dim=1) # (B, 15, act_dim)
        
        ##### Encoder #####
        z = self.encoder.encode_im(torch.cat(([o_history, o_target, o_goal]), 1)) # (B, 4, z_dim)
        z_action = self.encoder.encode_act(a)                                     # (B, 15, d) with 5 clean / 10 noisy

        losses["SIGReg"].append(torch.stack([self.sigreg(z[:, k].unsqueeze(0)) for k in range(4)]).mean()) # To do as leWAM but replace with other later
        # losses["SIGReg"].append(self.sigreg(z.transpose(0, 1)))

        ##### Predictor #####
        z_history = z[:, :2]
        z_target = z[:, 2].unsqueeze(1) # (B, 1, z_dim) to match the shape of z_pred at mse loss
        z_goal = z[:, 3].unsqueeze(1)   # (B, 1, z_dim) to match the shape for the cat at the MoT input
        z_next = self.next_state_query.expand(B, 1, -1)

        tokens = self.predictor_adaptor(z_history, z_goal, z_next, z_action)
        
        v_pred, z_pred = self.MoT(tokens, 
            self._attn_mask(history_pad), 
            tau, 
            h_norm
        ) # (B, 19, z_dim) -> (B, 10, act_dim), (B, 1, z_dim)

        # loss over prediction
        losses["MSE"].append(((z_pred - z_target) ** 2).mean())

        # loss over actions 
        err = ((v_pred - (actions - eps)) ** 2).mean(-1) * action_valid            # (B, 10)
        losses["MSE"].append((err.sum(-1) / action_valid.sum(-1).clamp(min=1)).mean())

        losses["zstd"] = z_target[:, 0].detach().std(0).mean() # to check if collapse during training

        losses["total"] = losses["MSE"][0] + losses["MSE"][1] + (losses["SIGReg"][0] * self.sigreg_coef) # total loss

        return losses

# ------------------------------- Generation -------------------------------

    @torch.no_grad()
    def generate(self, o_history, o_goal, h_norm, history_pad=None, inference_type="direct_policy"):
        """
        o_history : (B, 2, 3, 224, 224)  2 last images spaced by 5 steps
        o_goal    : (B, 1, 3, 224, 224)  goal image
        h_norm    : (B,)                 nb chunks needed to reach the goal_image
        inference_type : "direct_policy" | "best-of-K" | "gradien_based"
        -> (B, 5*n_steps, act_dim)
        """
        was_training = self.training
        self.eval()
        B = o_history.shape[0]
        if history_pad is None:
            history_pad = torch.zeros(B, 2, dtype=torch.bool, device=o_history.device)
        mask = self._attn_mask(history_pad)
        
        ##### Encoder #####
        z = self.encoder.encode_im(torch.cat(([o_history, o_goal]), 1))

        ##### Predictor reusable data #####
        z_history = z[:, :2]
        z_goal = z[:, 2].unsqueeze(1) # (B, 1, z_dim) to match the shape for the cat at the MoT input
        z_next = self.next_state_query.expand(B, 1, -1)


        match inference_type:
            case "direct_policy":
                image_tokens = self.predictor_adaptor(z_history, z_goal, z_next, None, type="images") # (B, 4, z_dim)
                x = self._direct_policy(B, image_tokens, mask, h_norm)[:, :5]
            case "best-of-K":
                _, x = self._best_of_k(B, z_history, z_goal, z_next, mask, h_norm)
            case "gradient_based":
                x = self._gradient_based(B, z_history, z_goal, z_next, mask, h_norm)
            case _:
                sys.exit('Choose one inference type')


        if was_training:
            self.train()
        
        return x
    
# ------------------------------- Generation methods -------------------------------

    def _direct_policy(self, B, image_tokens, mask, h_norm):
        """
        we sample a random action and upgrade it with {n_steps} euler steps on the flow policy
        (but not always the case https://arxiv.org/abs/2610.37891)

        B            : batch size
        image_tokens : (B, 4, z_dim)
        mask         : (B, 1, 19, 19)
        h_norm       : (B,)
        """
        x = torch.randn(B, 10, self.act_dim, device=mask.device)

        for i in range(self.n_euler_step):
            # Prepare data for MoT
            tau = torch.full((B,), i/self.n_euler_step, device=mask.device)
            a = torch.cat([torch.full((B, 5, self.act_dim), 0, device=mask.device), x], dim=1)
            z_action = self.encoder.encode_act(a)

            # fill only the missing part
            actions_tokens = self.predictor_adaptor(None, None, None, z_action, "actions")
            tokens = torch.cat([image_tokens, actions_tokens], dim=1)
            
            v_pred, _ = self.MoT(
                tokens,
                mask,
                tau,
                h_norm,
            )
            x = x + v_pred / self.n_euler_step
        return x # (B, 10, 2)

    def _rollout(self, B, z_history, z_goal, z_next, mask, h_norm, n_step):
        """
        it does {n_step} in latent environment, using direct_policy upgraded actions

        preds         : (n_step, B, z_dim) 
        actions       : (n_step, B, 10, act_dim)
        """
        preds = []
        actions = []
        zero_mask = self._attn_mask(torch.zeros(B, 2, dtype=torch.bool, device=z_history.device))
        for i in range(n_step):
            h = (h_norm - i / 10).clamp(min=0.1)
            m = mask if i == 0 else zero_mask # passed first iter theres aren't any padding left 

            image_tokens = self.predictor_adaptor(z_history, z_goal, z_next, None, "images")

            x = self._direct_policy(B, image_tokens, m, h)

            a_dyn = torch.cat([x[:, :5], torch.zeros_like(x)], dim=1)
            action_tokens = self.predictor_adaptor(None, None, None, self.encoder.encode_act(a_dyn), "actions")
            tokens = torch.cat([image_tokens, action_tokens], dim=1)

            _, z_pred = self.MoT(tokens, m, torch.ones(B, device=x.device), h)

            z_history = torch.cat([z_history[:, 1:], z_pred], dim=1) # (B*K, 2, z_dim)
            actions.append(x)
            preds.append(z_pred[:, 0])

        return preds, actions


    def _best_of_k(self, B, z_history, z_goal, z_next, mask, h_norm):
        """
        generate K possible actions using direct_policy and rollout to see which one is the "best of k"
        """
        K = self.k
        z_history = z_history.repeat_interleave(K, 0) # (B*K, 2, z_dim)
        z_goal = z_goal.repeat_interleave(K, 0) # (B*K, 1, z_dim)
        z_next = z_next.repeat_interleave(K, 0) # (B*K, 1, z_dim)
        mask = mask.repeat_interleave(K, 0)
        h_norm = h_norm.repeat_interleave(K, 0)

        steps_to_goal = (h_norm * 10).round().clamp(min=1).long() # (B*K,) chunks remainings
        n_steps = int(steps_to_goal.max()) # to farthest goal

        preds, actions = self._rollout(B*K, z_history, z_goal, z_next, mask, h_norm, n_steps)

        preds = torch.stack(preds, dim=1) # (B*K, n_steps, z_dim)
        z_at_goal = preds[torch.arange(preds.shape[0]), steps_to_goal - 1]
        cost = ((z_at_goal - z_goal[:, 0]) ** 2).mean(-1)

        best = cost.view(B, K).argmin(1)
        x = torch.cat([a[:, :5] for a in actions], 1).view(B, K, 5*n_steps, -1)
        return preds, x[torch.arange(B, device=x.device), best]   # (B, 5*n_steps, act_dim)


    def _gradient_based(self, B, z_history, z_goal, z_next, mask, h_norm):
        """
        generate K possible actions using direct_policy and optimize them {gd_step} times with gradient descent and select the best
        """
        K = self.k
        z_history = z_history.repeat_interleave(K, 0) # (B*K, 2, z_dim)
        z_goal = z_goal.repeat_interleave(K, 0) # (B*K, 1, z_dim)
        z_next = z_next.repeat_interleave(K, 0) # (B*K, 1, z_dim)
        mask = mask.repeat_interleave(K, 0)
        h_norm = h_norm.repeat_interleave(K, 0)

        steps_to_goal = (h_norm * 10).round().clamp(min=1).long() # (B*K)
        n_steps = int(steps_to_goal.max()) # to farthest goal

        _, actions = self._rollout(B * K, z_history, z_goal, z_next, mask, h_norm, n_steps) # simulate {n_steps} steps
        actions = torch.stack(actions, 0) # (n_step, B*K, 10, act_dim)

        actions_cpy = actions.clone()
        best_actions = actions.clone()
        best_cost = torch.full((B * K,), float("inf"), device=actions.device)

        z_hist_cpy = z_history # z_history will move with doing rollout so we keep a backup
        # build a zero-padding mask because after the first step there is no padding
        zero_mask = self._attn_mask(torch.zeros(B*K, 2, dtype=torch.bool, device=z_history.device)) 
        optimizer = optim.Adam([actions], self.lr_for_gd)
        cost = None 

        with torch.enable_grad():
            actions.requires_grad = True
            for i in range(self.n_gd_step + 1): # 1 step further to eval
                z_history = z_hist_cpy
                z_pred = None
                preds = [] # keep the predictions to get the cost
                for j in range(n_steps):
                    h = (h_norm - j / 10).clamp(min=0.1)
                    m = mask if j == 0 else zero_mask # passed first iter theres aren't any padding left 

                    # cat predicted actions to zeros, because we only need the next chunk
                    action = torch.cat([actions[j, :, :5], torch.zeros(B * K, 10, self.act_dim, device=actions.device)], dim=1)
                    z_action = self.encoder.encode_act(action)

                    tokens = self.predictor_adaptor(z_history, z_goal, z_next, z_action, "full")

                    _, z_pred = self.MoT(tokens, m, torch.ones(B * K, device=action.device), h) # (B*K, 1, z_dim)
                    preds.append(z_pred[:, 0])
                    z_history = torch.cat([z_history[:, 1:], z_pred], dim=1) # go to next state (t-5 = t and t = pred_t)
                
                preds = torch.stack(preds, dim=1) # (B*K, n_steps, z_dim)
                z_at_goal = preds[torch.arange(preds.shape[0]), steps_to_goal - 1] # select last z for each B*K
                cost = ((z_at_goal - z_goal[:, 0]) ** 2).mean(-1)

                # keep the best iterate of each candidate (i = 0 is the sampled plan)
                with torch.no_grad():
                    better = cost < best_cost
                    best_actions[:, better] = actions[:, better]
                    best_cost[better] = cost[better]

                if i == self.n_gd_step :
                    break # we don't do gradient iteration at last : only for eval
                # if we do a last iteration, cost would be outdated as the actions changed

                loss = cost.sum() + self.lambda_prox * ((actions - actions_cpy) ** 2).sum() # as in LeWAM
                actions.grad = torch.autograd.grad(loss, actions)[0] # not to pollute model during training
                nn.utils.clip_grad_norm_([actions], 10.0)
                optimizer.step()

                with torch.no_grad():
                    actions.clamp_(-self.a_max, self.a_max) # as in LeWAM

        # select only the best of Ks
        best = best_cost.view(B, K).argmin(1)
        x = torch.cat([a[:, :5] for a in best_actions], 1).view(B, K, 5*n_steps, -1) 
        return x[torch.arange(B, device=actions.device), best]
            