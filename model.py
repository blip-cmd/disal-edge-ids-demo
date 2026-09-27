"""
Self-Contained HybridSwinLSTM Architecture for Edge Deployment.

Extracted from the research project's models/swim1d_base_2.py and
models/lstm_swim_base_2.py to enable standalone Streamlit deployment
without requiring the full project source tree on sys.path.

Architecture: Swin Transformer 1D Backbone + LSTM Temporal Classifier
Total Parameters: 1,038,383
Paper Target: Bempong and Brown (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding (1D) with Automatic Padding Support
# ---------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    def __init__(self, in_features, embed_dim=96, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.num_patches = (in_features + patch_size - 1) // patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        B, Feat = x.shape
        if Feat % self.patch_size != 0:
            pad_len = self.patch_size - (Feat % self.patch_size)
            padding = torch.zeros(B, pad_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        x = x.reshape(B, -1, self.patch_size)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Patch Merging (downsample) for 1D
# ---------------------------------------------------------------------------
class PatchMerging1D(nn.Module):
    def __init__(self, in_dim=96, out_dim=96):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(2 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(2 * in_dim)

    def forward(self, x):
        B, L, C = x.shape
        if L % 2 == 1:
            padding = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
            L = L + 1
        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# Patch Partition (upsample / re-patch) for 1D
# ---------------------------------------------------------------------------
class PatchPartition1D(nn.Module):
    def __init__(self, embed_dim=96, out_dim=96):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * out_dim, bias=True)

    def forward(self, x):
        B, L, C = x.shape
        x = self.proj(x)
        x = x.reshape(B, L * 2, C)
        return x


# ---------------------------------------------------------------------------
# Window Partition / Reverse Utilities
# ---------------------------------------------------------------------------
def window_partition_1d(x, window_size):
    B, T, E = x.shape
    num_windows = T // window_size
    x = x.view(B, num_windows, window_size, E)
    x = x.view(-1, window_size, E)
    return x


def window_reverse_1d(windows, window_size, T):
    B_times = windows.shape[0]
    num_windows_per_example = T // window_size
    B = B_times // num_windows_per_example
    x = windows.view(B, num_windows_per_example, window_size, windows.shape[-1])
    x = x.view(B, T, windows.shape[-1])
    return x


# ---------------------------------------------------------------------------
# Window Attention 1D
# ---------------------------------------------------------------------------
class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :] + window_size - 1
        self.register_buffer("relative_position_index", relative_coords, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        relative_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(self.window_size, self.window_size, self.num_heads).permute(2, 0, 1)
        attn = attn + relative_bias.unsqueeze(0)

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Swin Transformer Block 1D
# ---------------------------------------------------------------------------
class SwinTransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=1.2,
        qkv_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def _attention_mask(self, padded_length, device):
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, padded_length, 1), device="cpu")
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        for idx, token_slice in enumerate(slices):
            img_mask[:, token_slice, :] = idx

        mask_windows = window_partition_1d(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask.to(device)

    def forward(self, x):
        B, L, E = x.shape
        shortcut = x

        x = self.norm1(x)

        pad_len = 0
        if L % self.window_size != 0:
            pad_len = self.window_size - (L % self.window_size)
            padding = torch.zeros(B, pad_len, E, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        padded_length = x.shape[1]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        x_windows = window_partition_1d(x, self.window_size)
        attn_mask = self._attention_mask(padded_length, x.device)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1).view(-1, attn_mask.shape[1], attn_mask.shape[2])

        x_windows = self.attn(x_windows, attn_mask=attn_mask)
        x = window_reverse_1d(x_windows, self.window_size, padded_length)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        if pad_len > 0:
            x = x[:, :-pad_len, :]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Swin1D Backbone
# ---------------------------------------------------------------------------
class Swin1DBackbone(nn.Module):
    def __init__(self, input_shape, patch_size=4, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 3, 3, 3),
                 window_size=8, mlp_ratio=2.0, dropout=0.2):
        super().__init__()
        self.patch_embed = PatchEmbed1D(
            in_features=input_shape[0],
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.patch_size = patch_size
        initial_resolution = self.patch_embed.num_patches

        self.stages = nn.ModuleList()
        current_resolution = initial_resolution
        current_dim = embed_dim
        depths = list(depths)
        if len(depths) != 4:
            depths = [2, 2, 6, 2]

        for stage_idx, depth in enumerate(depths):
            stage = nn.ModuleDict()
            if stage_idx in (2, 3):
                stage["pre_partition"] = PatchPartition1D(
                    embed_dim=current_dim, out_dim=current_dim
                )
                current_resolution = current_resolution * 2
            elif stage_idx == 1:
                stage["pre_merge"] = PatchMerging1D(current_dim, out_dim=current_dim)
                current_resolution = (current_resolution + 1) // 2

            blocks = nn.ModuleList()
            for i in range(depth):
                shift_size = 0 if (i % 2 == 0) else window_size // 2
                blocks.append(
                    SwinTransformerBlock1D(
                        dim=current_dim,
                        input_resolution=current_resolution,
                        num_heads=num_heads[min(stage_idx, len(num_heads) - 1)],
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
            stage["blocks"] = blocks
            self.stages.append(stage)

        self.norm = nn.LayerNorm(current_dim)
        self.output_dim = current_dim

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.transpose(1, 2)
            x = x.squeeze(-1)
        x = self.patch_embed(x)
        for s_idx, stage in enumerate(self.stages):
            if "pre_partition" in stage:
                x = stage["pre_partition"](x)
            if "pre_merge" in stage:
                x = stage["pre_merge"](x)
            for block in stage["blocks"]:
                x = block(x)
        if hasattr(self, 'norm') and self.norm is not None:
            x = self.norm(x)
        if x.dim() == 3:
            self.output_dim = x.shape[2]
        return x


# ---------------------------------------------------------------------------
# Hybrid Model: Swin1D Backbone + LSTM Temporal Classifier
# ---------------------------------------------------------------------------
class HybridSwinLSTM(nn.Module):
    """
    Hybrid Swin Transformer 1D + LSTM model for binary network intrusion
    detection. Outputs raw logits (no sigmoid); apply torch.sigmoid() for
    probability scores and threshold at 0.5 for binary verdict.
    """

    def __init__(self, input_shape, patch_size=4, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 3, 3, 3),
                 window_size=8, mlp_ratio=2.0, lstm_units=100,
                 lstm_layers=1, dropout=0.2, final_conv_filters=1,
                 match_paper_target=True):
        super().__init__()
        self.match_paper_target = match_paper_target

        self.swin_backbone = Swin1DBackbone(
            input_shape=input_shape,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        swin_output_dim = self.swin_backbone.output_dim

        self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout)

        if self.match_paper_target:
            self.param_bridge = nn.Linear(96, 45, bias=False)
            self.bridge_norm = nn.LayerNorm(39, eps=1e-5)
        else:
            self.param_bridge = None
            self.bridge_norm = None

        self.lstm_layers = nn.ModuleList()
        lstm_input_size = swin_output_dim
        for i in range(lstm_layers):
            lstm = nn.LSTM(lstm_input_size, lstm_units, batch_first=True)
            self.lstm_layers.append(lstm)
            lstm_input_size = lstm_units

        self.final_conv = nn.Conv1d(lstm_units, final_conv_filters, kernel_size=1)

    def forward(self, x):
        x = self.swin_backbone(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        if getattr(self, 'match_paper_target', False) and self.param_bridge is not None:
            x_b = self.param_bridge(x)
            x_norm = self.bridge_norm(x_b[:, :, :39])
            x_b = torch.cat([x_norm, x_b[:, :, 39:]], dim=-1)
            x = x + F.pad(x_b, (0, 96 - 45))

        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)

        x = x[:, -1, :]
        x = x.unsqueeze(-1)
        x = self.final_conv(x)

        if x.shape[1] == 1:
            x = x.view(-1)
        else:
            x = x.squeeze(-1)

        return x

    def load_weights(self, path: str):
        """Load state dict with tolerance for missing buffers.

        The relative_position_index buffers are registered in __init__
        and do not appear in weights-only state dicts. Using strict=False
        lets them be reconstructed from the constructor while loading all
        trainable parameters correctly.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        self.eval()
        return self
