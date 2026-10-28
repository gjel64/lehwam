"""
Train LeWAM on an h5 dataset.

Runs are configured under configs/train/base.yaml and a protocol config (gr.yaml, tc.yaml), and can be overridden
on the command line:
  python scripts/train_lewam.py --config-name gr dataset_path=<train h5> run_name=<run>
  python scripts/train_lewam.py --config-name tc dataset_path=<train h5> run_name=<run> \
      data.views=[pixels,pixels_r0eih,pixels_r1eih]
  python scripts/train_lewam.py --config-name hwam dataset_path=<train h5> run_name=<run> train.init_from=<lewam run>

Check docs/DATA.md for the data sources and the README for the paper's runs.
"""

import json
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, RandomSampler

from lewam.models.encoders import ResNetEncoder
from lewam.models.hwam import HWAM
from lewam.models.lewam import LeWAM, build_model
from lewam.models.module import SIGReg
from lewam.train.datasets import TrajectoryDataset, get_start_indices, load_frames, read_actions
from lewam.train.utils import _IMG_MEAN, _IMG_STD, durable_sync, subgoal_lambda

torch.backends.cudnn.benchmark = True


def model_config(cfg: DictConfig, action_raw_dim: int) -> dict:
    """Return model config"""
    model = cfg.model
    views = list(cfg.data.views)
    goal_views = list(cfg.data.goal_views) or views[:1]
    assert all(view in views for view in goal_views), f"data.goal_views {goal_views} must be a subset of {views}"
    return dict(
        fs=cfg.data.frameskip, action_raw_dim=action_raw_dim, img_size=cfg.data.img_size,
        encoder_size=model.encoder_size, encoder_backbone=model.encoder_backbone, encoder_ckpt=None,
        crop_size=model.crop_size,
        z_dim=model.z_dim, proj_hidden=model.proj_hidden, d_model=model.d_model, n_heads=model.n_heads,
        depth=model.depth, dropout=model.dropout, n_flow_steps=model.n_flow_steps,
        num_actions_pred=model.num_actions_pred, num_states_pred=model.num_states_pred,
        history_len=model.history_len, history_stride=model.history_stride, goal_type=model.goal_type,
        H_max=model.H_max, action_head=model.action_head, action_head_init=model.action_head_init, state_head=model.state_head,
        flow_alpha_act=float(model.flow_alpha_act), flow_alpha_state=float(model.flow_alpha_state),
        num_views=len(views), views=views, goal_views=goal_views,
        goal_view_indices=[views.index(view) for view in goal_views],
        sigreg_grouped=bool(model.sigreg_grouped),
        # no regularizer -> no projection module
        sigreg_proj_dim=model.sigreg_proj_dim if cfg.train.w_reg > 0 else 0,
        subgoal_dim=cfg.hwam.dims[0] if cfg.hwam.enabled and cfg.hwam.subgoal_token else 0,
    )


def make_loader(dataset: TrajectoryDataset, batch_size: int, steps_per_epoch: int, **loader_kwargs) -> DataLoader:
    """Return DataLoader of TrajectoryDataset"""
    num_samples = max(batch_size, len(dataset))
    if steps_per_epoch:
        num_samples = min(num_samples, steps_per_epoch * batch_size)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=RandomSampler(dataset, num_samples=num_samples), **loader_kwargs
    )


def build_param_groups(model: LeWAM, idm_head: nn.Module | None, hwam: HWAM | None, train_cfg: DictConfig) -> list[dict]:
    """Return AdamW param groups of the trainable parameters: encoder / dynamics (the state stream) / policy (+ IDM head)
    / hwam."""
    group_lr = {"encoder": train_cfg.encoder_lr, "policy": train_cfg.policy_lr, "dynamics": train_cfg.dynamics_lr, "hwam": None}
    params = {"encoder": [], "dynamics": [], "policy": [], "hwam": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[model.param_group_of(name)].append(param)
    if idm_head is not None:
        params["policy"] += list(idm_head.parameters())
    if hwam is not None:
        params["hwam"] = list(hwam.parameters())

    groups = [dict(params=params[name], lr=train_cfg.lr if group_lr[name] is None else group_lr[name], name=name,
                   n=sum(p.numel() for p in params[name]))
              for name in ("encoder", "policy", "dynamics", "hwam") if params[name]]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for module in (idm_head, hwam):
        if module is not None:
            n_params += sum(p.numel() for p in module.parameters())

    assert sum(group["n"] for group in groups) == n_params
    return groups


def lr_scale(epoch: int, warmup_epochs: int, epochs: int) -> float:
    """Return a learning-rate multiplier for epoch, with linear warmup and cosine decay to 0."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def encode_frames(model: LeWAM, frames, batch_size: int, device: str) -> torch.Tensor:
    """Encode every frame once with the encoder in eval mode (train.cache_latents): frames from `load_frames`
    -> (N, [views,] z_dim) latents on the CPU"""
    model.encoder.eval()
    latents = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        pixels = (batch.flatten(0, -4).to(device).float() / 255.0 - _IMG_MEAN.to(device)) / _IMG_STD.to(device)
        latents.append(model.encode(pixels).view(*batch.shape[:-3], -1).cpu())
    return torch.cat(latents)


def reshape_view_major(latents: torch.Tensor, B: int, n_frames: int, n_views: int) -> torch.Tensor:
    """Reshape (B * n_frames * views, z) encoder output -> (B, views * n_frames, z), camera 0 first."""
    z_dim = latents.shape[-1]
    return latents.reshape(B, n_frames, n_views, z_dim).transpose(1, 2).reshape(B, n_views * n_frames, z_dim)



def sigreg_loss(
    sigreg: SIGReg, model: LeWAM, z_history: torch.Tensor, state_target: torch.Tensor, z_goal: torch.Tensor | None, grouped: bool
) -> torch.Tensor:
    """
    SIGReg anti-collapse loss on the encoder latents

    Args:
        sigreg (SIGReg): the regularizer
        model (LeWAM): the model
        z_history (torch.Tensor): (B, views*history_len, z_dim) history latents
        state_target (torch.Tensor): (B, views*num_states_pred, z_dim) next-state latents
        z_goal (torch.Tensor | None): (B, z_dim) or (B, goal views, z_dim) goal latents
        grouped (bool): one loss per latent group (each history frame, state and goal view), averaged; else
            one loss over the newest history frame and the states pooled

    Returns:
        loss (torch.Tensor): scalar loss
    """
    project = model.sigreg_proj if model.sigreg_proj_dim > 0 else (lambda x: x)
    if not grouped:
        pooled = torch.cat([z_history[:, -1], state_target.reshape(-1, state_target.shape[-1])])
        return sigreg(project(pooled).unsqueeze(0))

    groups = [z_history[:, k] for k in range(z_history.shape[1])]
    groups += [state_target[:, q] for q in range(state_target.shape[1])]
    if z_goal is not None:
        groups += [z_goal] if z_goal.dim() == 2 else [z_goal[:, g] for g in range(z_goal.shape[1])]

    return torch.stack([sigreg(project(group).unsqueeze(0)) for group in groups]).mean()


def hwam_loss(
    hwam: HWAM, model: LeWAM, z_strip: torch.Tensor, strip_actions: torch.Tensor, z_goal: torch.Tensor | None,
    h_norm: torch.Tensor | None, cfg: DictConfig, lam: float
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    """
    HWAM loss of one batch, and the subgoals of the LeWAM policy with hwam.subgoal_token

    Args:
        hwam (HWAM): the levels above LeWAM
        model (LeWAM): the model
        z_strip (torch.Tensor): (B, strip_past + strip_ahead, z0_dim) level-0 latents
        strip_actions (torch.Tensor): (B, strip_ahead * frameskip, action_dim) z-scored actions from the current step
        z_goal (torch.Tensor | None): (B, z_dim) or (B, goal views, z_dim) goal latents
        h_norm (torch.Tensor | None): (B,) normalized goal horizon
        cfg (DictConfig): the run config
        lam (float): probability of a subgoal planned by level 1 instead of the hindsight subgoal

    Returns:
        loss (torch.Tensor): scalar loss
        loss_terms (dict[str, float]): the logged loss per term, the unweighted loss as "hwam"
        z_subgoal (torch.Tensor | None): (B, dims[0]) subgoals, None without subgoal token
    """
    levels = hwam.elevate(z_strip)
    loss, loss_terms = hwam.loss(levels, strip_actions, cfg.train.w_reg)
    loss_terms["hwam"] = loss.item()
    if not model.subgoal_dim:
        return loss, loss_terms, None

    # hindsight: the next real level-1 latent; generated: the first latent of a level-1 plan to the goal
    current = hwam.current_index(1)
    z_subgoal = levels[0][:, current + 1].detach().clone()
    generated = (torch.rand(z_subgoal.shape[0], device=z_subgoal.device) < lam).nonzero()[:, 0]
    if generated.numel():
        history = levels[0][generated, current - hwam.history_len + 1:current + 1].detach()
        goal = hwam.goals(z_goal[generated].flatten(1))[0].detach()
        goal_steps = (h_norm[generated] * cfg.model.H_max).round().long()
        steps_to_goal = (goal_steps + hwam.strides[0] - 1) // hwam.strides[0]
        init = history.new_zeros(generated.numel(), int(steps_to_goal.max()), *hwam.cond_shapes[0])
        _, latents = hwam.plan(1, history, hwam.goal_cost(goal, steps_to_goal), init, mode="grad",
                               grad_steps=hwam.infer_steps, u_reg_weight=hwam.u_reg_weight)
        z_subgoal[generated] = latents[:, 0].to(z_subgoal.dtype)
    return loss, loss_terms, z_subgoal


def run_batch(
    batch: tuple, model: LeWAM, idm_head: nn.Module | None, hwam: HWAM | None, sigreg: SIGReg, cfg: DictConfig,
    device: str, train: bool, w_hwam: float = 1.0, lam: float = 0.0
) -> tuple[torch.Tensor, dict[str, float], int]:
    """
    Encode one batch and compute the training loss

    Args:
        batch (tuple): a collated TrajectoryDataset batch, of frames or of cached latents
        model (LeWAM): the model
        idm_head (nn.Module | None): the inverse-dynamics head, if train.w_idm > 0
        hwam (HWAM | None): the levels above LeWAM, if hwam.enabled
        sigreg (SIGReg): the regularizer
        cfg (DictConfig): the run config
        device (str): "cuda" or "cpu"
        train (bool): training (goal dropout, encoder checkpointing) or validation
        w_hwam (float): weight of the HWAM loss
        lam (float): probability of a generated subgoal

    Returns:
        loss (torch.Tensor): scalar total loss
        loss_terms (dict[str, float]): the logged loss per term
        B (int): the batch size
    """
    train_cfg = cfg.train
    goal_conditioning = cfg.model.goal_type != "none"
    (history_frames, history_pad, action_target, action_valid, state_frames,
     state_valid, goal_frames, h_norm, *strip) = (x.to(device, non_blocking=True) for x in batch)

    action_target = action_target.float()
    B, n_history = history_frames.shape[:2]
    n_states_pred = cfg.model.num_states_pred
    cached = history_frames.is_floating_point()     # train.cache_latents: the frames are latents
    item_shape = history_frames.shape[-1:] if cached else history_frames.shape[-3:]
    n_views = history_frames.shape[2] if history_frames.dim() == 3 + len(item_shape) else 1
    if cfg.model.goal_type != "sampled":
        h_norm = None

    frames = [history_frames.reshape(B * n_history * n_views, *item_shape),
              state_frames.reshape(B * n_states_pred * n_views, *item_shape)]
    n_goal_views = len(model.goal_view_indices)
    if goal_conditioning:
        goal_views = goal_frames[:, model.goal_view_indices] if goal_frames.dim() == 2 + len(item_shape) else goal_frames[:, None]
        frames.append(goal_views.reshape(B * n_goal_views, *item_shape))
    if strip:
        strip_frames, strip_actions = strip
        n_strip = strip_frames.shape[1]
        frames.append(strip_frames.reshape(B * n_strip * n_views, *item_shape))

    if cached:
        z = torch.cat(frames)
    else:
        pixels = (torch.cat(frames).float() / 255.0 - _IMG_MEAN.to(device)) / _IMG_STD.to(device)
        if train and train_cfg.encoder_checkpoint_chunks > 1:
            z = torch.cat([checkpoint(model.encode, chunk, use_reentrant=False)
                           for chunk in pixels.chunk(train_cfg.encoder_checkpoint_chunks)])
        else:
            z = model.encode(pixels)

    z_history = reshape_view_major(z[:B * n_history * n_views], B, n_history, n_views)
    state_target = reshape_view_major(z[B * n_history * n_views:B * (n_history + n_states_pred) * n_views], B, n_states_pred, n_views)
    if n_views > 1:
        state_valid = state_valid.repeat(1, n_views)
    z_goal = goal_keep = None
    n_lewam = B * (n_history + n_states_pred) * n_views
    if goal_conditioning:
        z_goal = z[n_lewam:n_lewam + B * n_goal_views].reshape(B, n_goal_views, -1)
        n_lewam += B * n_goal_views
        if n_goal_views == 1:
            z_goal = z_goal[:, 0]
        if train and train_cfg.p_drop_goal > 0:
            goal_keep = torch.rand(B, device=device) >= train_cfg.p_drop_goal

    loss, loss_terms, z_subgoal = 0.0, {}, None
    if hwam is not None:
        # level 0 of HWAM: the latents of the goal views, concatenated
        z_strip = z[n_lewam:].reshape(B, n_strip, n_views, -1)[:, :, model.goal_view_indices].flatten(2)
        loss, loss_terms, z_subgoal = hwam_loss(hwam, model, z_strip, strip_actions, z_goal, h_norm, cfg, lam)
        loss = w_hwam * loss
        if train_cfg.freeze_lewam:
            return loss, loss_terms, B

    loss_action, loss_state = model.loss(z_history, history_pad, action_target, action_valid, state_target,
                                         state_valid, z_goal=z_goal, h_norm=h_norm, goal_keep=goal_keep, z_subgoal=z_subgoal)
    loss = loss + train_cfg.w_act * loss_action + train_cfg.w_dyn * loss_state
    loss_terms.update({"act": loss_action.item(), "state": loss_state.item()})

    if train_cfg.w_reg > 0:
        with torch.autocast(device, enabled=False):   # bf16 rounds the statistic and its gradient
            loss_reg = sigreg_loss(sigreg, model, z_history.float(), state_target.float(),
                                   None if z_goal is None else z_goal.float(), cfg.model.sigreg_grouped)
        loss = loss + train_cfg.w_reg * loss_reg
        loss_terms["reg"] = loss_reg.item()

    if idm_head is not None:
        idm_input = torch.cat([z_history[:, n_history - 1], state_target[:, 0]], dim=-1)
        loss_idm = F.mse_loss(idm_head(idm_input), action_target[:, :cfg.data.frameskip].reshape(B, -1))
        loss = loss + train_cfg.w_idm * loss_idm
        loss_terms["idm"] = loss_idm.item()
    if n_states_pred:
        loss_terms["zstd"] = state_target.float().reshape(-1, state_target.shape[-1]).std(0).mean().item()

    return loss, loss_terms, B


def accumulate(store: dict[str, float], loss_terms: dict[str, float], n: int) -> None:
    for key, value in loss_terms.items():
        store[key] = store.get(key, 0.0) + value * n


def format_stats(store: dict[str, float], n: float) -> str:
    keys = [key for key in ("act", "state", "reg", "zstd") if key in store] + [key for key in store if key.startswith("h")]
    return " ".join(f"{key}={store[key] / max(n, 1):.5f}" for key in keys)


@hydra.main(version_base="1.3", config_path="../configs/train", config_name=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg, model_cfg, train_cfg, hwam_cfg = cfg.data, cfg.model, cfg.train, cfg.hwam

    assert model_cfg.num_actions_pred >= model_cfg.num_states_pred * data_cfg.frameskip, \
        "num_actions_pred must cover the actions of every predicted state (num_states_pred * frameskip)"
    assert 1 <= model_cfg.history_stride <= data_cfg.frameskip, (
        f"model.history_stride {model_cfg.history_stride} must be in 1..{data_cfg.frameskip}; "
        f"wider history spacing than one dynamics step would not match the spacing of imagined rollouts"
    )
    assert train_cfg.precision in ("fp32", "bf16"), train_cfg.precision
    if hwam_cfg.enabled:
        assert train_cfg.freeze_encoder or not train_cfg.cache_latents, "train.cache_latents needs train.freeze_encoder"
        assert not (hwam_cfg.subgoal_token and train_cfg.freeze_lewam), "hwam.subgoal_token trains LeWAM: train.freeze_lewam=false"
        assert train_cfg.subgoal_source == "hindsight" or model_cfg.goal_type == "sampled", \
            "generated subgoals plan to the sampled goal: model.goal_type=sampled"

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(cfg.dataset_path, list(data_cfg.views), data_cfg.source, data_cfg.decomp_dir, data_cfg.load_in_ram)
    actions, (action_mean, action_std) = read_actions(cfg.dataset_path)
    frame_starts, ep_starts, frames_to_terminal = get_start_indices(cfg.dataset_path, data_cfg.frameskip, data_cfg.terminal_state)
    assert frames.shape[-1] == data_cfg.img_size, f"frames are {frames.shape[-1]}px, data.img_size is {data_cfg.img_size}"
    n_starts = frame_starts.shape[0]
    print(f"[lewam] data_source={data_cfg.source} frames={tuple(frames.shape)} starts={n_starts} "
          f"raw_dim={actions.shape[1]} history_stride={model_cfg.history_stride} goal_type={model_cfg.goal_type}",
          flush=True)

    lewam_cfg = model_config(cfg, actions.shape[1])
    model = build_model(lewam_cfg).to(device)
    if train_cfg.compile:
        assert isinstance(model.encoder, ResNetEncoder), "train.compile supports the resnet encoders"
        model.encoder.trunk.compile()   # in place: the state_dict keys are unchanged
        model.encoder.trunk_chunks = train_cfg.compile_chunks

    hwam = None
    if hwam_cfg.enabled:
        if train_cfg.init_from:
            # a released checkpoint has no subgoal token
            state = torch.load(run_dir.parent / train_cfg.init_from / "lewam_best.pt", map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected and all(key.startswith("subgoal_proj.") for key in missing), (missing, unexpected)
        model.requires_grad_(not train_cfg.freeze_lewam)
        model.encoder.requires_grad_(not train_cfg.freeze_encoder)
        if train_cfg.cache_latents:
            frames = encode_frames(model, frames, train_cfg.batch_size, device)
        lewam_cfg["hwam"] = OmegaConf.to_container(hwam_cfg, resolve=True)
        hwam = HWAM(lewam_cfg).to(device)
        print(f"[hwam] params={sum(p.numel() for p in hwam.parameters()) / 1e6:.2f}M init_from={train_cfg.init_from} "
              f"freeze_encoder={train_cfg.freeze_encoder} freeze_lewam={train_cfg.freeze_lewam} "
              f"cache_latents={train_cfg.cache_latents}", flush=True)

    possible_starts = torch.randperm(n_starts, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = int(round((1 - data_cfg.train_split) * possible_starts.numel()))
    loader_kwargs = dict(num_workers=data_cfg.num_workers, pin_memory=True)
    if data_cfg.num_workers:
        loader_kwargs.update(prefetch_factor=data_cfg.prefetch_factor, persistent_workers=True)
    train_loader, val_loader = [
        make_loader(
            TrajectoryDataset(
                frames, actions, frame_starts, ep_starts, frames_to_terminal, starts,
                model_cfg.history_len, model_cfg.history_stride, model_cfg.num_actions_pred,
                model_cfg.num_states_pred, data_cfg.frameskip, model_cfg.goal_type, model_cfg.H_max,
                *((hwam.strip_past, hwam.strip_ahead) if hwam is not None else ())
            ),
            train_cfg.batch_size, train_cfg.steps_per_epoch, **loader_kwargs
        ) for starts in (possible_starts[n_val:], possible_starts[:n_val])
    ]

    (run_dir / "lewam_config.json").write_text(
        json.dumps({**lewam_cfg, "action_mean": action_mean, "action_std": action_std}, indent=1)
    )
    (run_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"[lewam] params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  w_reg={train_cfg.w_reg}", flush=True)

    idm_head = None
    if train_cfg.w_idm > 0:
        # recovers the first action chunk from (z_t, z_{t+fs}), shaping the encoder toward action-aware latents
        assert model_cfg.num_states_pred >= 1, "train.w_idm needs a state slot (model.num_states_pred >= 1)"
        idm_head = nn.Sequential(
            nn.Linear(2 * model_cfg.z_dim, 512), nn.SiLU(),
            nn.Linear(512, data_cfg.frameskip * actions.shape[1])
        ).to(device)

    param_groups = build_param_groups(model, idm_head, hwam, train_cfg)
    optimizer = torch.optim.AdamW(param_groups, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    print("[lewam] lr groups: " + ", ".join(f"{group['name']} lr={group['lr']:.2e} n={group['n']}" for group in param_groups), flush=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(lr_scale, warmup_epochs=train_cfg.warmup_epochs, epochs=train_cfg.epochs)
    )

    start_epoch, best_val = 0, float("inf")
    full_path = run_dir / "lewam_full.pt"
    if cfg.resume and full_path.exists():
        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        if idm_head is not None:
            idm_head.load_state_dict(state["idm_head"])
        if hwam is not None:
            hwam.load_state_dict(state["hwam"])
        print(f"[lewam] resumed at epoch {start_epoch} best_val={best_val:.5f}", flush=True)

    sigreg = SIGReg().to(device)
    use_bf16 = device == "cuda" and train_cfg.precision == "bf16"
    autocast = partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext
    print(f"[lewam] precision={train_cfg.precision}", flush=True)
    # the best-checkpoint metric: the val action loss, or the HWAM loss when LeWAM is frozen
    best_metric = "hwam" if hwam is not None and train_cfg.freeze_lewam else "act"

    def save(suffix: str) -> list[Path]:
        """Save lewam_<suffix>.pt, and hwam_<suffix>.pt with HWAM"""
        modules = {"lewam": model} if hwam is None else {"lewam": model, "hwam": hwam}
        for prefix, module in modules.items():
            torch.save(module.state_dict(), run_dir / f"{prefix}_{suffix}.pt")
        return [run_dir / f"{prefix}_{suffix}.pt" for prefix in modules]

    for epoch in range(start_epoch, train_cfg.epochs):
        start_time = time.time()
        model.train()
        hwam_kwargs = {}
        if hwam is not None:
            hwam.train()
            warmup = min(1.0, (epoch + 1) / train_cfg.hwam_warmup_epochs) if train_cfg.hwam_warmup_epochs else 1.0
            lam = subgoal_lambda(train_cfg.subgoal_source, train_cfg.hwam_schedule, epoch / max(1, train_cfg.epochs - 1),
                                 train_cfg.lam_max, train_cfg.lam_center, train_cfg.lam_sharpness)
            hwam_kwargs = dict(w_hwam=train_cfg.w_hwam * warmup, lam=lam)
        train_stats, train_n = {}, 0
        for batch in train_loader:
            with autocast():
                loss, loss_terms, n = run_batch(batch, model, idm_head, hwam, sigreg, cfg, device, train=True, **hwam_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if hwam is not None:
                torch.nn.utils.clip_grad_norm_(hwam.parameters(), 1.0)
            optimizer.step()
            accumulate(train_stats, loss_terms, n)
            train_n += n
        scheduler.step()

        model.eval()
        if hwam is not None:
            hwam.eval()
        val_stats, val_n = {}, 0
        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    loss, loss_terms, n = run_batch(batch, model, idm_head, hwam, sigreg, cfg, device, train=False, **hwam_kwargs)
                accumulate(val_stats, loss_terms, n)
                val_n += n
        val_metric = val_stats.get(best_metric, 0.0) / max(val_n, 1)
        print(f"[lewam] ep {epoch + 1}/{train_cfg.epochs}  train[{format_stats(train_stats, train_n)}]  "
              f"val[{format_stats(val_stats, val_n)}]  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{time.time() - start_time:.1f}s", flush=True)

        # the std of the LeWAM latents and of each HWAM level
        val_zstd = min((val_stats[key] for key in val_stats if key.endswith("zstd")), default=float("inf")) / max(val_n, 1)
        if train_cfg.zstd_floor > 0 and epoch + 1 >= 5 and val_zstd < train_cfg.zstd_floor:
            (run_dir / "collapse_killed").write_text(
                f"epoch {epoch + 1} val zstd {val_zstd:.6f} < floor {train_cfg.zstd_floor}\n")
            print(f"[lewam] COLLAPSE_KILL zstd={val_zstd:.6f} < {train_cfg.zstd_floor} at ep {epoch + 1}", flush=True)
            break

        latest_files = save("latest")
        full_state = dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, best_val=best_val
        )
        if idm_head is not None:
            full_state["idm_head"] = idm_head.state_dict()
        if hwam is not None:
            full_state["hwam"] = hwam.state_dict()
        torch.save(full_state, full_path)

        files = [run_dir / "lewam_config.json", run_dir / "train_config.yaml", *latest_files, full_path]
        if val_metric < best_val:
            best_val = val_metric
            files += save("best")
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            files += save(f"ep{epoch + 1}")
        if cfg.ckpt_sync_dir:
            durable_sync(files, cfg.ckpt_sync_dir)

    print(f"[lewam] DONE best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    main()
