import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .modules import MLP


# ------------------------------------------------ Utils ------------------------------------------------


def spatial_softmax(fmap):
    """(N, 32, h, w) -> (N, 64) : mean position (x, y) of cards in [-1, 1]"""
    N, K, H, W = fmap.shape
    attn = fmap.flatten(2).softmax(-1)                  # (N, 32, h*w) : distrib per card
    xs = torch.linspace(-1.0, 1.0, W, device=fmap.device)
    ys = torch.linspace(-1.0, 1.0, H, device=fmap.device)
    grid_x = xs.repeat(H)
    grid_y = ys.repeat_interleave(W)
    return torch.cat([(attn * grid_x).sum(-1), (attn * grid_y).sum(-1)], dim=-1)   # (N, 64)

def batchnorm_to_groupnorm(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Replace every BatchNorm2d in `module` (in place, recursively) with GroupNorm(num_groups)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            batchnorm_to_groupnorm(child, num_groups)
    return module


# ------------------------------------------------ Encoder ------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, emb_dim=192, act_dim=2, crop_size=202, hidden_coef=2):
        super().__init__()
        # normally conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # so we remove the last two layers (avgpool and fc) and replace all bn with groupnorm
        trunk = nn.Sequential(*list(torchvision.models.resnet18(weights=None).children())[:-2])
        # as in Diffusion Policy’s ResNet18 and as in leWAM
        self.resnet18dp = batchnorm_to_groupnorm(trunk)
        self.conv = nn.Conv2d(512, 32, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_coef*emb_dim),
            nn.BatchNorm1d(hidden_coef*emb_dim), # for SIGReg
            nn.GELU(),
            nn.Linear(hidden_coef*emb_dim, emb_dim)
        )
        self.act_proj = nn.Linear(act_dim, emb_dim)

        self.crop_size = crop_size
        
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    
    def encode_im(self, pixels : torch.Tensor):
        """
        x : (B, i, 3, 224, 224)
        """
        B, i, _, _, _ = pixels.shape
        x = pixels.flatten(0, 1).float() / 255.0 # (B*i, 3, 224, 224)
        x = (x - self.im_mean) / self.im_std

        if self.crop_size:
            # arrange before croping 
            x = self.crop(x) # (B*i, 3, 202, 202)

        x = self.resnet18dp(x)            # (B*i, 512, 7, 7)
        x = self.conv(x)                  # (B*i, 32, 7, 7)
        x = spatial_softmax(x)            # (B*i, 64) 
        z = self.proj(x)                  # (B*i, 192)

        return z.view(B, i, -1)

    def encode_act(self, a):
        """
        a : (B, 10, act_dim)
        """
        z_action = self.act_proj(a) # (B, 10, 192)
        return z_action

    def crop(self, pixels: torch.Tensor) -> torch.Tensor:
        """Crop (N, 3, H, W) to crop_size: a random offset per image in training, the center in eval."""
        N, C, H, W = pixels.shape
        s = self.crop_size
        if self.training:
            top = torch.randint(0, H - s + 1, (N,), device=pixels.device)
            left = torch.randint(0, W - s + 1, (N,), device=pixels.device)
        else:
            top = pixels.new_full((N,), (H - s) // 2, dtype=torch.long)
            left = pixels.new_full((N,), (W - s) // 2, dtype=torch.long)
        # index into unfolded views: gathering with broadcast indices materialized ~2GB per batch
        windows = pixels.unfold(2, s, 1).unfold(3, s, 1)            # (N, C, H-s+1, W-s+1, s, s) view
        return windows[torch.arange(N, device=pixels.device), :, top, left]



# ------------------------------------------------ HEncoder ------------------------------------------------


class HEncoder(nn.Module):
    def __init__(self, emb_dim, below_dim, dropout_p, depth):
        super().__init__()
        self.d = emb_dim

        self.mlp = MLP(below_dim, emb_dim, dropout_p=dropout_p, depth=depth, norm_fn=nn.BatchNorm1d)


    def forward(self, token):
        """
        token : (B, T, below_D)
        normally T is 1
        """
        B, T, _ = token.shape

        x = token.flatten(0, 1) # (B*T, D_below)
        return self.mlp(x).reshape(B, T, self.d) # (B, T, D)

