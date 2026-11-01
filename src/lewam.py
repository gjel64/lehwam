import torch
import torch.nn as nn
import types

from .encoder import Encoder
from .predictor import MoT
from .modules import SIGReg

# ------------------------------------------------ LeWAM ------------------------------------------------


class leWAM(nn.Module):
    def __init__(self, emb_dim, act_dim, device, sigreg_coef=0.04):
        super().__init__()
        self.encoder = Encoder(emb_dim, act_dim)
        self.MoT = MoT(emb_dim, act_dim)

        self.next_state_query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

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

        self.cfg = types.SimpleNamespace() # To simulate a class
        self.cfg.n_step = 8
        self.cfg.act_dim = act_dim

        # SIGReg
        self.sigreg = SIGReg()
        self.sigreg_coef = sigreg_coef

        # MoT adaptator
        self.hist_proj = nn.Linear(emb_dim, emb_dim)
        self.goal_proj = nn.Linear(emb_dim, emb_dim)
        self.hist_pos = nn.Parameter(torch.zeros(1, 2, emb_dim))     # zeros as LeWAM
        self.action_pos = nn.Parameter(torch.zeros(1, 10, emb_dim))
        self.state_type = nn.Parameter(torch.randn(3, emb_dim))      # 0 history, 1 next state, 2 goal
        self.action_type = nn.Parameter(torch.randn(2, emb_dim))     # 0 clean, 1 noisy

    def predictor_adaptor(self, z_history, z_goal, z_next, z_action, type="full"):
        """ 
        adapt the zs to MoT by adding position, proj (z->usable_z), type
        takes the 'raw' embeddings of an encoder and adapt it to predictor
        -> very usefull w/ a pretrained encoder  
        (B, 2, z_dim), (B, 1, z_dim), (B, 1, z_dim), (B, 15, z_dim) -> (B, 19, z_dim) tokens for the MoT
        """
        # the type is used because at generate, it calc actions everytimes but images once
        match type:
            case "full":
                h = self.hist_proj(z_history) + self.hist_pos + self.state_type[0]
                n = z_next + self.state_type[1]
                g = self.goal_proj(z_goal) + self.state_type[2]
                clean = z_action[:, :5] + self.action_pos[:, :5] + self.action_type[0]
                noisy = z_action[:, 5:] + self.action_pos + self.action_type[1]
                return torch.cat([h, n, g, clean, noisy], dim=1)
            case "actions":
                clean = z_action[:, :5] + self.action_pos[:, :5] + self.action_type[0]
                noisy = z_action[:, 5:] + self.action_pos + self.action_type[1]
                return torch.cat([clean, noisy], dim=1)
            case "images":
                h = self.hist_proj(z_history) + self.hist_pos + self.state_type[0]
                n = z_next + self.state_type[1]
                g = self.goal_proj(z_goal) + self.state_type[2]
                return torch.cat([h, n, g], dim=1)



    def forward(self, o_history, o_target, o_goal, actions, h_norm):
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

        losses["SIGReg"].append(self.sigreg(z.transpose(0, 1)))

        ##### Predictor #####
        z_history = z[:, :2]
        z_target = z[:, 2].unsqueeze(1) # (B, 1, z_dim) to match the shape of z_pred at mse loss
        z_goal = z[:, 3].unsqueeze(1)   # (B, 1, z_dim) to match the shape for the cat at the MoT input
        z_next = self.next_state_query.expand(B, 1, -1)

        tokens = self.predictor_adaptor(z_history, z_goal, z_next, z_action)
        
        v_pred, z_pred = self.MoT(
            tokens,
            self.mask,
            tau,
            h_norm
        ) # (B, 19, z_dim) -> (B, 10, act_dim), (B, 1, z_dim)
        
        losses["MSE"].append(((z_pred - z_target) ** 2).mean())
        losses["MSE"].append(((v_pred - (actions - eps)) ** 2).mean())

        losses["total"] = losses["MSE"][0] + losses["MSE"][1] + (losses["SIGReg"][0] * self.sigreg_coef)

        return losses


    @torch.no_grad()
    def generate(self, o_history, o_goal, h_norm):
        """
        o_history : (B, 2, 3, 224, 224)  2 last images spaced by 5 steps
        o_goal    : (B, 1, 3, 224, 224)  goal image
        h_norm    : (B,)                 nb chunks needed to reach the goal_image
        -> (B, 10, act_dim)              actions standardized (centré réduit)
        """
        was_training = self.training
        self.eval()
        B = o_history.shape[0]
        
        ##### Encoder #####
        z = self.encoder.encode_im(torch.cat(([o_history, o_goal]), 1))

        x = torch.randn(B, 10, self.cfg.act_dim, device=o_history.device)
        a = torch.cat([torch.full((B, 5, self.cfg.act_dim), 0, device=o_history.device), x], dim=1)

        ##### Predictor #####
        z_history = z[:, :2]
        z_goal = z[:, 2].unsqueeze(1) # (B, 1, z_dim) to match the shape for the cat at the MoT input
        z_next = self.next_state_query.expand(B, 1, -1)

        # false z_action not to repet every time
        image_tokens = self.predictor_adaptor(z_history, z_goal, z_next, None, type="images")

        for i in range(self.cfg.n_step):
            # Prepare data for MoT
            tau = torch.full((B,), i/self.cfg.n_step, device=o_history.device)
            a = torch.cat([torch.full((B, 5, self.cfg.act_dim), 0, device=o_history.device), x], dim=1)
            z_action = self.encoder.encode_act(a)

            # fill only the missing part
            actions_tokens = self.predictor_adaptor(None, None, None, z_action, "actions")
            tokens = torch.cat([image_tokens, actions_tokens], dim=1)
            
            v_pred, _ = self.MoT(
                tokens,
                self.mask,
                tau,
                h_norm
            )
            x = x + v_pred / self.cfg.n_step

        if was_training:
            self.train()
        
        return x

