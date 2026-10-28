"""HWAM (IDEA.md): the training loss of every option, the inference, the planning, the subgoal schedule, the strip and
the LeWAM subgoal token. Tiny models on the CPU."""
import pytest
import torch

from lewam.models.hwam import HWAM
from lewam.models.lewam import build_model
from lewam.train.datasets import TrajectoryDataset
from lewam.train.utils import subgoal_lambda


def make_hwam(**overrides) -> HWAM:
    hwam_cfg = dict(num_levels=2, strides=[2, 3], dims=[8, 8], history_len=2, elevator="transformer", elevator_depth=1,
                    predictor_depth=1, action_cond="latent", u_dim=4, u_reg="sigreg", u_reg_weight=0.1, infer_steps=3,
                    infer_lr=0.1, infer_optimizer="adam", infer_grad="detach")
    return HWAM(dict(z_dim=12, fs=2, action_raw_dim=2, n_heads=2, hwam={**hwam_cfg, **overrides}))


@pytest.mark.parametrize("overrides", [
    {}, {"elevator": "mlp_window"}, {"elevator": "mlp_frame"}, {"action_cond": "actions"}, {"u_reg": "l2"},
    {"u_reg": "none", "infer_optimizer": "sgd"}, {"infer_grad": "last_step"}, {"infer_grad": "unroll"},
])
def test_loss_trains_every_level(overrides):
    torch.manual_seed(0)
    hwam = make_hwam(**overrides)
    strip = torch.randn(16, hwam.strip_past + hwam.strip_ahead, 12, requires_grad=True)
    levels = hwam.elevate(strip)
    assert [z.shape[1] for z in levels] == [(hwam.strip_past + hwam.strip_ahead) // span for span in hwam.spans]
    loss, loss_terms = hwam.loss(levels, torch.randn(16, hwam.strip_ahead * 2, 2), w_reg=0.1)
    loss.backward()
    assert torch.isfinite(loss) and {"h1_dyn", "h2_reg", "h2_zstd"} <= loss_terms.keys()
    assert strip.grad is not None
    assert all(p.grad is not None for p in hwam.parameters())


def test_inference_lowers_the_energy():
    torch.manual_seed(0)
    hwam = make_hwam(infer_steps=20, u_reg_weight=0.0)
    history, target = torch.randn(32, 2, 8), torch.randn(32, 8)
    u = hwam.infer(1, history, target)
    assert hwam.energy(1, history, u, target).mean() < hwam.energy(1, history, torch.zeros_like(u), target).mean()


@pytest.mark.parametrize("mode", ["cem", "grad", "cem_grad"])
def test_plan_lowers_the_goal_cost(mode):
    torch.manual_seed(0)
    hwam = make_hwam()
    history, goal, steps_to_goal = torch.randn(4, 2, 8), torch.randn(4, 8), torch.tensor([1, 2, 3, 3])
    init = torch.zeros(4, 3, 1, 4)
    goal_cost = hwam.goal_cost(goal, steps_to_goal)
    plan, latents = hwam.plan(1, history, goal_cost, init, mode=mode, num_proposals=16, u_reg_weight=0.0)
    assert plan.shape == init.shape and latents.shape == (4, 3, 8)
    assert (goal_cost(latents[:, None]) <= goal_cost(hwam.rollout(1, history, init)[:, None]) + 1e-6).all()


def test_consistency_window_starts_after_the_elapsed_real_latents():
    hwam = make_hwam(elevator="mlp_window")
    real, imagined, target = torch.randn(2, 2, 12), torch.randn(2, 4, 12), torch.randn(2, 8)
    cost = hwam.consistency(1, imagined, target, real, elapsed=torch.tensor([0, 1]))
    windows = torch.stack([imagined[0, :2], torch.stack([real[1, -1], imagined[1, 0]])])
    assert torch.allclose(cost, ((hwam.elevators[0](windows) - target) ** 2).mean(-1))


def test_subgoal_lambda():
    sigmoid = dict(source="scheduled", schedule="sigmoid", lam_max=0.8, center=0.5, sharpness=10.0)
    assert subgoal_lambda(progress=0.0, **sigmoid) == pytest.approx(0.0)
    assert subgoal_lambda(progress=0.5, **sigmoid) == pytest.approx(0.4)
    assert subgoal_lambda(progress=1.0, **sigmoid) == pytest.approx(0.8)
    assert subgoal_lambda("scheduled", "linear", 0.5, 0.8, 0.5, 10.0) == pytest.approx(0.4)
    assert subgoal_lambda("hindsight", "sigmoid", 0.7, 0.8, 0.5, 10.0) == 0.0
    assert subgoal_lambda("generated", "sigmoid", 0.1, 0.8, 0.5, 10.0) == 1.0


def test_strip_is_clamped_to_the_episode():
    latents = torch.arange(20.0)[:, None]          # one 20-frame episode of cached latents, latent = frame index
    actions = torch.ones(20, 2).half()
    dataset = TrajectoryDataset(latents, actions, torch.tensor([3]), torch.tensor([0]), torch.tensor([16]),
                                torch.tensor([0]), 2, 2, 4, 1, 2, "none", 10, strip_past=4, strip_ahead=2)
    *_, strip, strip_actions = dataset[0]
    assert strip[:, 0].tolist() == [0, 0, 1, 3, 5, 7]
    assert strip_actions.shape == (4, 2)


def test_lewam_subgoal_token_and_mse_start():
    torch.manual_seed(0)
    model = build_model(dict(action_raw_dim=2, img_size=64, crop_size=58, encoder_size="tiny", goal_type="sampled",
                             depth=1, subgoal_dim=8, action_head_init="mse"))
    B = 3
    z_history, history_pad = torch.randn(B, 2, 192), torch.zeros(B, 2, dtype=torch.bool)
    goal = dict(z_goal=torch.randn(B, 192), h_norm=torch.rand(B))
    loss_action, loss_state = model.loss(z_history, history_pad, torch.randn(B, 10, 2), torch.ones(B, 10),
                                         torch.randn(B, 1, 192), torch.ones(B, 1), z_subgoal=torch.randn(B, 8), **goal)
    (loss_action + loss_state).backward()
    assert model.subgoal_proj.weight.grad is not None and model.action_query.grad is not None
    with torch.no_grad():
        torch.manual_seed(1)
        with_subgoal = model.predict_actions(z_history, history_pad, z_subgoal=torch.randn(B, 8), **goal)
        torch.manual_seed(1)
        without_subgoal = model.predict_actions(z_history, history_pad, **goal)
    assert with_subgoal.shape == (B, 10, 2) and not torch.equal(with_subgoal, without_subgoal)
