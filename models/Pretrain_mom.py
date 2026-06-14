import torch
import torch.nn as nn
from einops import rearrange, repeat
import os
import ot
os.environ["PYTHONIOENCODING"] = "utf-8"
from einops import repeat
import lightning.pytorch as pl
from transformers import AutoTokenizer, BertTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from models.med import BertModel
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat
import numpy as np
import re
import json
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torchmetrics.classification import MulticlassAccuracy
from peft import get_peft_model, LoraConfig, TaskType
from transformers import logging
logging.set_verbosity_error()

FINDING_COLS = [
    ("consolidation", "consolidation_progression"),
    ("edema", "edema_progression"),
    ("pleural_effusion", "pleural_effusion_progression"),
    ("pneumonia", "pneumonia_progression"),
    ("pneumothorax", "pneumothorax_progression"),
]

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, key_mask=None, register_hook=False):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # bs, head, C,C  12,393,393
        # mask 
        if N != 197  : # bs,393
            key_mask=key_mask.to(attn.device)
            patnum= q.size(2)
            attn = attn.masked_fill(key_mask.unsqueeze(1).repeat(1,patnum,1).unsqueeze(1).repeat(1,self.num_heads,1,1), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if register_hook:
            self.save_attention_map(attn)
            if attn.requires_grad:
                attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.mlp_l = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.norm2_l = norm_layer(dim)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)
            # TODO mlp_l

    def forward(self, x, view_type=None,key_mask=None, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), key_mask=key_mask,
                               register_hook=register_hook))
        if (x.shape[1]==197):
            if (view_type == "frontal"):
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else: 
                x = x + self.drop_path(self.mlp_l(self.norm2_l(x)))
        else:
            fout = self.drop_path(self.mlp(self.norm2(x[:, :197,:])))
            lout = self.drop_path(self.mlp_l(self.norm2_l(x[:,197:,:])))
            x = x + torch.cat((fout, lout), dim=1)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(
                    use_grad_checkpointing and i >= depth-ckpt_layer)
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=11): 
        B = x.shape[0]
        x = self.patch_embed(x)  # 39,196,768
        cls_tokens = self.cls_token.expand(B, -1, -1) # 39,1,768
        x = torch.cat((cls_tokens, x), dim=1)

        L, d = x.shape[1],x.shape[2]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        view_type = "frontal"
        for i, blk in enumerate(self.blocks):
            x = blk(x, view_type, register_blk == i)
        x = self.norm(x)
           
        return x
    

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(
            stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(
                            block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(
                            block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(
                            block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(
        w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size,
                                        orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' %
              (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


class GatingLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, residual, x):
        gating_weights = torch.sigmoid(self.linear(torch.cat([residual, x], dim=-1)))
        x = gating_weights * x  # element-wise product
        x = residual + x
        return x
    
# Spatial-Temporal Gating
class STGLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0., 
                 is_gating=False, 
                 is_temporal_first=False,
                 no_mlp=False,
                 no_spatial=False,
                 no_temporal=False,
                no_spatial_gating=False,
                no_temporal_gating=False,
                no_mlp_gating=False,
                save_spatial_attn=False,
                save_temporal_attn=False
                 ):
        super().__init__()

        self.is_gating = is_gating
        self.is_temporal_first = is_temporal_first
        self.no_mlp = no_mlp
        self.no_spatial = no_spatial
        self.no_temporal = no_temporal
        self.no_spatial_gating=no_spatial_gating
        self.no_temporal_gating=no_temporal_gating
        self.no_mlp_gating=no_mlp_gating
        # 新增
        self.save_spatial_attn = save_spatial_attn
        self.last_spatial_attn = None
        self.save_temporal_attn = save_temporal_attn
        self.last_temporal_attn = None
        
        if not self.is_temporal_first: # STG
            if not self.no_spatial:
                self.pre_norm1 = nn.LayerNorm(hidden_dim) 
                self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                if self.is_gating: 
                    self.gating1 = GatingLayer(hidden_dim)

            if not self.no_temporal:
                self.pre_norm2 = nn.LayerNorm(hidden_dim) 
                self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                if self.is_gating:
                    self.gating2 = GatingLayer(hidden_dim)
                    
        else: # TSG
            if not self.no_temporal:
                self.pre_norm1 = nn.LayerNorm(hidden_dim) 
                self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                if self.is_gating: 
                    self.gating1 = GatingLayer(hidden_dim)

            if not self.no_spatial:
                self.pre_norm2 = nn.LayerNorm(hidden_dim) 
                self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                if self.is_gating: 
                    self.gating2 = GatingLayer(hidden_dim)
        
        if not self.no_mlp:
            self.pre_norm3 = nn.LayerNorm(hidden_dim)
            self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, int(mlp_ratio * hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(mlp_ratio * hidden_dim), hidden_dim),
                    nn.Dropout(dropout)
                )
            if self.is_gating:
                self.gating3 = GatingLayer(hidden_dim)

    def forward(self, x):
        # Get shape, (batch, num_frames, num_patch_tokens, hidden_dim)
        B, T, L, D = x.shape
                    
        if not self.is_temporal_first:
            # Spatial attention
            x = rearrange(x, 'B T L D -> (B T) L D')
            residual = x
            if not self.no_spatial:
                x, attn = self.spatial_attn(x, x, x, need_weights=True,
                        average_attn_weights=False)
                if self.save_spatial_attn:
                    self.last_spatial_attn = attn.detach()
                if self.is_gating:
                    x = self.gating1(residual, x)
                else:
                    x = residual + x
                    
            # Temporal attention
            x = rearrange(x, '(B T) L D -> (B L) T D', T=T)
            residual = x
            if not self.no_temporal:
                x, attn = self.temporal_attn(x, x, x, need_weights=True,
                        average_attn_weights=False)
                if self.save_temporal_attn:
                    self.last_temporal_attn = attn.detach()
                if self.is_gating:
                    x = self.gating2(residual, x)
                else:
                    x = residual + x
        else:
            # Temporal attention
            x = rearrange(x, 'B T L D -> (B L) T D')
            residual = x
            if not self.no_temporal:
                x, attn = self.temporal_attn(x, x, x, need_weights=True,
                        average_attn_weights=False)
                if self.save_temporal_attn:
                    self.last_temporal_attn = attn.detach()
                if self.is_gating:
                    x = self.gating2(residual, x)
                else:
                    x = residual + x
            
            # Spatial attention
            x = rearrange(x, '(B L) T D -> (B T) L D', L=L)
            residual = x
            if not self.no_spatial:
                x, attn = self.spatial_attn(x, x, x, need_weights=True,
                        average_attn_weights=False)
                if self.save_spatial_attn:
                    self.last_spatial_attn = attn.detach()
                if self.is_gating:
                    x = self.gating1(residual, x)
                else:
                    x = residual + x
            
        # MLP
        if not self.is_temporal_first:
            x = rearrange(x, '(B L) T D -> B T L D', L=L)
        else:
            x = rearrange(x, '(B T) L D -> B T L D', T=T)
        
        if not self.no_mlp:
            x = self.pre_norm3(x)
            residual = x
            x = self.mlp(x)
            if self.is_gating:
                x = self.gating3(residual, x)
            else:
                x = residual + x
        return x

# ST-Gating
class STG(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0., 
                 is_gating=False, 
                 is_temporal_first=True, 
                 no_spatial=False, 
                 no_temporal=False,
                 no_mlp=False, 
                 use_pos_patch_level=False, 
                 use_pos_frame_level=False, 
                 use_pos_absolute=False,
                 num_frames=3, 
                 num_patch_tokens=196,
                 use_xavier_init=False, 
                 no_spatial_gating=False,
                 no_temporal_gating=False,
                 no_mlp_gating=False,
                 save_spatial_attn=False,
                 save_temporal_attn=False
):
        super().__init__()
        
        self.use_pos_patch_level = use_pos_patch_level
        self.use_pos_frame_level = use_pos_frame_level
        self.use_pos_absolute = use_pos_absolute
        self.num_frames = num_frames
        
        if self.use_pos_patch_level:
            # input.shape = (batch, num_frames, num_patch_tokens, hidden_dim)
            # num_patch_tokens = 256 + 1, including cls token
            # each patch token has a position embedding
            if self.use_pos_absolute:
                # absolute, sinusoidal position encoding
                self.pos_embedding = self._get_sinusoidal_position_encoding(num_frames*num_patch_tokens, hidden_dim)
                self.pos_embedding = rearrange(self.pos_embedding, '(T L) D -> T L D', L=num_patch_tokens)
                self.pos_embedding = self.pos_embedding.unsqueeze(0)
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patch_tokens, hidden_dim))
        elif self.use_pos_frame_level: 
            # frame-level, temporal position embedding
            # each frame has a position embedding
            # all patch tokens in the same frame share the same position embedding
            if self.use_pos_absolute:
                # absolute, sinusoidal position encoding
                self.pos_embedding = self._get_sinusoidal_position_encoding(num_frames, hidden_dim)
                self.pos_embedding = self.pos_embedding.unsqueeze(0).unsqueeze(2)
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim))        
        
        self.layers = nn.ModuleList([
                STGLayer(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, 
                            is_gating=is_gating, is_temporal_first=is_temporal_first, 
                            no_mlp=no_mlp,
                            no_spatial=no_spatial,
                            no_temporal=no_temporal,
                            no_spatial_gating=no_spatial_gating,
                            no_temporal_gating=no_temporal_gating,
                            no_mlp_gating=no_mlp_gating,
                            save_spatial_attn=save_spatial_attn,
                            save_temporal_attn=save_temporal_attn
                            ) 
                for _ in range(num_layers)
            ])
        
        # 增加一个可学习的感知帧
        self.perception_frame = nn.Parameter(
            torch.randn(1, num_patch_tokens, hidden_dim), 
            requires_grad=True
        )

        self.use_xavier_init = use_xavier_init
        if self.use_xavier_init:
            self.apply(self._xavier_normal_init)

    def _xavier_normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.0)

    def _get_sinusoidal_position_encoding(self, max_len, dim):
        if dim % 2 != 0: 
            raise ValueError(f"Cannot create sinusoidal position encoding for odd dimension: {dim}")
        
        with torch.no_grad():
            position_encoding = torch.zeros(max_len, dim) # max_len, dimension
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape = (max_len, 1)

            _2i = torch.arange(0, dim, 2, dtype=torch.float)
            position_encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / dim)))
            position_encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / dim)))

        return position_encoding

    def forward(self, prior_x, curr_x):
        bz, L, D = prior_x.shape
        percep_frame = repeat(self.perception_frame, '1 L D -> B L D', B=bz) 
        x = torch.cat([
            prior_x.unsqueeze(1),
            percep_frame.unsqueeze(1),
            curr_x.unsqueeze(1)
        ], dim=1)  # B, T, L, D 

        if self.use_pos_patch_level or self.use_pos_frame_level:
            if self.use_pos_patch_level:
                assert x.shape[2] == self.pos_embedding.shape[2], f"Number of patch tokens should be equal to the number of position embeddings. {x.shape[2]} != {self.pos_embedding.shape[2]}"
            if self.use_pos_absolute:
                self.pos_embedding = self.pos_embedding.to(x.device)
            x = x + self.pos_embedding[:, :self.num_frames, :L]

        # STG layers
        for layer in self.layers:
            x = layer(x)
        # x shape: bs x T x L x dim
        return x
    
    def forward_with_attn(self, prior_x, curr_x, layer_idx=2):
        B, L, D = prior_x.shape
        percep_frame = repeat(self.perception_frame, '1 L D -> B L D', B=B)
        percep_frame = percep_frame    
        x = torch.cat([
            prior_x.unsqueeze(1),
            percep_frame.unsqueeze(1),
            curr_x.unsqueeze(1)
        ], dim=1)  # B, T, L, D 
               
        x = x + self.pos_embedding

        spatial_attn = None
        temporal_attn = None
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "save_spatial_attn"):
                layer.save_spatial_attn = (i == layer_idx)
            if hasattr(layer, "save_temporal_attn"):
                layer.save_temporal_attn = (i == layer_idx)
            x = layer(x)

            if i == layer_idx and getattr(layer, "last_spatial_attn", None) is not None:
                spatial_attn = layer.last_spatial_attn  # [B*T, num_heads, L, L]

            if i == layer_idx and getattr(layer, "last_temporal_attn", None) is not None:
                temporal_attn = layer.last_temporal_attn  # [B*L, num_heads, L, L]

        return x, spatial_attn, temporal_attn


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)

class BertEncoder(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = False):
        super(BertEncoder, self).__init__()
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        # self.max_sent_num = 10

        self.model = BertModel.from_pretrained(
            "/bulk/huz5/Test/MedST/medst/emilyalsentzer/Bio_ClinicalBERT"
        )
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("/bulk/huz5/Test/MedST/medst/emilyalsentzer/Bio_ClinicalBERT")

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        #state_dict = torch.load("/bulk/huz5/Test/MedST/medst/medst.ckpt", map_location=device)['state_dict']
        #print('Loaded pretrained visual encoder weights.')
        #print('loaded keys:', visual_state_dict.keys())
        #new_state_dict = {}
        #for k, v in state_dict.items():
        #    if k.startswith("text_encoder_q.model."):
        #        new_k = k[len("text_encoder_q.model."):]
        #        new_state_dict[new_k] = v
        # 打印调用的参数
        #missing_keys, unexpected_keys = self.model.load_state_dict(state_dict=new_state_dict, strict=True)
        #print('missing_keys:', missing_keys)
        #loaded_param_names = set(new_state_dict.keys()) - set(unexpected_keys)
        #print('loaded_param_names:', loaded_param_names)

        #for param in self.model.parameters():
        #    param.requires_grad = False
        #self.model.eval()   


    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)
        
        return agg_embs_batch, sentences, last_atten_pt

    def forward(self, ids, attn_mask, token_type):
        outputs = self.model(ids, attn_mask, token_type,
                             return_dict=True, mode="text")

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)
        # print('all_feat shape', all_feat.shape) bs x 1 x seq_len x dim

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, ids, last_layer_attn)
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in ids]

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        return report_feat, word_feat, last_atten_pt, sents
    
class ImageEncoder(nn.Module):
    def __init__(self,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.output_dim = output_dim
        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        image_size = 224

        self.visual_encoder, vision_width = create_vit(
            'base', image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.feature_dim = vision_width

        visual_state_dict = torch.load("/bulk/huz5/Test/MedST/medst/medst.ckpt", map_location=device)['state_dict']
        #print('Loaded pretrained visual encoder weights.')
        #print('loaded keys:', visual_state_dict.keys())
        new_state_dict = {}
        for k, v in visual_state_dict.items():
            if k.startswith("img_encoder_q.model."):
                new_k = k[len("img_encoder_q.model."):]
                new_state_dict[new_k] = v
        # 打印调用的参数
        missing_keys, unexpected_keys = self.visual_encoder.load_state_dict(state_dict=new_state_dict, strict=True)
        #print('missing_keys:', missing_keys)
        loaded_param_names = set(new_state_dict.keys()) - set(unexpected_keys)
        #print('loaded_param_names:', loaded_param_names)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()   

    def forward(self, x):
        img_feat = self.visual_encoder(x)
        return img_feat[:, 1:].contiguous()
        
class OT_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau

    def OT(self, weight1, weight2):
        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            print("cost_map shape: ", self.cost_map.shape)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            print("cost_map shape: ", self.cost_map.shape)
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double(), reg=self.ot_reg,reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow # (N, M)
            dist = torch.sum(dist) # (1,) float
            return flow, dist
        
        else:
            raise NotImplementedError

    def forward(self,x,y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sum_dist = 0.0 
        for i in range(x.shape[0]):
            _, dist = self.OT(x[i], y[i])
            sum_dist += dist
        return sum_dist / x.shape[0]
    
class Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.visual_encoder = ImageEncoder()

        self.tokenizer = BertTokenizer.from_pretrained(
            "/bulk/huz5/Test/MedST/medst/emilyalsentzer/Bio_ClinicalBERT"
        )
        self.text_model = BertEncoder().to(device)

        
        self.end_sym = args.end_sym
        self.STG = STG(num_layers=3, is_gating=False, is_temporal_first=True, use_pos_patch_level=True,
                       save_spatial_attn=True, save_temporal_attn=True)
        self.best_train_loss = float('inf')
        self.local_embed_t = LocalEmbedding()
        self.local_embed_i = LocalEmbedding()
        self.proj_progress = nn.Linear(768, 768)
        self.proj_reverse = nn.Linear(768, 768)
        self.recon_loss = nn.MSELoss()
        self.temperature = 0.07
        self.momentum = 0.995

        # get momentum encoder
        self.text_model_m = BertEncoder().to(device)
        self.STG_m = STG(num_layers=3, is_gating=False, is_temporal_first=True, use_pos_patch_level=True)
        self.local_embed_t_m = LocalEmbedding()
        self.local_embed_i_m = LocalEmbedding()
        self.model_pairs = [[self.text_model, self.text_model_m],
                            [self.STG, self.STG_m],
                            [self.local_embed_t, self.local_embed_t_m],
                            [self.local_embed_i, self.local_embed_i_m],
                           ]
        
        self.copy_params()
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    def get_text_feature(self, text):
        text_tokens = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=256,
        )
        ids = text_tokens["input_ids"].to(device)
        token_type = text_tokens["token_type_ids"].to(device)
        attention_mask = text_tokens["attention_mask"].to(device)
        report_feat, word_feat, last_atten_pt, sents = self.text_model(ids, attention_mask, token_type)
        word_emb = self.local_embed_t(word_feat)
        word_emb = F.normalize(word_emb, dim=-1)
        
        return word_emb, sents
    
    def get_text_feature_m(self, text):
        text_tokens = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=256,
        )
        ids = text_tokens["input_ids"].to(device)
        token_type = text_tokens["token_type_ids"].to(device)
        attention_mask = text_tokens["attention_mask"].to(device)
        report_feat_m, word_feat_m, last_atten_pt_m, sents_m = self.text_model(ids, attention_mask, token_type)
        word_emb_m = self.local_embed_t_m(word_feat_m)
        word_emb_m = F.normalize(word_emb_m, dim=-1)
        return word_emb_m, sents_m
    
    def get_token_similarity(self, img_feat1, img_feat2, text_feat1, text_feat2, sents1, sents2):
        patch_emb = torch.cat([img_feat1, img_feat2], dim=0)
        word_emb = torch.cat([text_feat1, text_feat2], dim=0)

        sents = sents1 + sents2
        bz = patch_emb.shape[0]
    
        pad_mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            patch_emb) 

        bs, patch_token_num, dim1 = patch_emb.shape
        _, word_token_num, dim2  = word_emb.shape
        patch_emb = torch.reshape(patch_emb, (bs*patch_token_num, dim1))
        word_emb = torch.reshape(word_emb, (bs*word_token_num, dim2)).transpose(1,0)
        # print("patch_emb_q:",patch_emb_q.shape)
        # print("word_emb_q:",word_emb_q.shape)
        tokens_sim_matrix = patch_emb @ word_emb # (bs x patch_token_num) x (bs x word_token_num)
        # print("tokens_sim_matrix:",tokens_sim_matrix.shape)
        # mask for useful word tokens
        word_tokens_mask = 1 - pad_mask
        word_tokens_mask = word_tokens_mask.flatten().unsqueeze(0) # 1 x (bs x word_token_num)
        tokens_sim_matrix = tokens_sim_matrix * word_tokens_mask # (bs x patch_token_num) x (bs x word_token_num)
        # get patch_sim_matrix
        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, word_token_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs 
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*word_token_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, word_token_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs 
        # batch["cap_lens"] : bs,
        word_sim_matrix = word_sim_matrix / bz  # bs x bs的shape
        patch_sim_matrix = patch_sim_matrix / self.temperature
        word_sim_matrix = word_sim_matrix / self.temperature
        
        return patch_sim_matrix, word_sim_matrix
    
    def get_single_token_similarity(self, patch_emb, word_emb, sents):
    
        bz = patch_emb.shape[0]
    
        pad_mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            patch_emb) 

        bs, patch_token_num, dim1 = patch_emb.shape
        _, word_token_num, dim2  = word_emb.shape
        patch_emb = torch.reshape(patch_emb, (bs*patch_token_num, dim1))
        word_emb = torch.reshape(word_emb, (bs*word_token_num, dim2)).transpose(1,0)
        # print("patch_emb_q:",patch_emb_q.shape)
        # print("word_emb_q:",word_emb_q.shape)
        tokens_sim_matrix = patch_emb @ word_emb # (bs x patch_token_num) x (bs x word_token_num)
        # print("tokens_sim_matrix:",tokens_sim_matrix.shape)
        # mask for useful word tokens
        word_tokens_mask = 1 - pad_mask
        word_tokens_mask = word_tokens_mask.flatten().unsqueeze(0) # 1 x (bs x word_token_num)
        tokens_sim_matrix = tokens_sim_matrix * word_tokens_mask # (bs x patch_token_num) x (bs x word_token_num)
        # get patch_sim_matrix
        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, word_token_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs 
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*word_token_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, word_token_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs 
        # batch["cap_lens"] : bs,
        word_sim_matrix = word_sim_matrix / bz  # bs x bs的shape
        patch_sim_matrix = patch_sim_matrix / self.temperature
        word_sim_matrix = word_sim_matrix / self.temperature
        
        return patch_sim_matrix, word_sim_matrix

    def rec_loss(self, I, Ihat):
        B, L, D = I.shape
        I_flat = I.reshape(B * L, D)
        Ihat_flat = Ihat.reshape(B * L, D)

        I_flat = F.normalize(I_flat, dim=-1)
        Ihat_flat = F.normalize(Ihat_flat, dim=-1)
        logits = I_flat @ Ihat_flat.t()        
        logits = logits / self.temperature
        targets = torch.arange(B * L, device=I.device)
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def get_image_feature(self, patch_feat):
        patch_emb = self.local_embed_i(patch_feat)
        patch_emb = F.normalize(patch_emb, dim=-1)
        return patch_emb
    
    
    def get_image_feature_m(self, patch_feat):
        patch_emb_m = self.local_embed_i_m(patch_feat)
        patch_emb_m = F.normalize(patch_emb_m, dim=-1)
        return patch_emb_m
    
    def forward_all_feat(self, prior_imgs, current_imgs, progression_text):
        patch_embed = self.visual_encoder(current_imgs)  # B x L x D
        prior_patch_embed = self.visual_encoder(prior_imgs)
        
        feats = self.STG(prior_patch_embed, patch_embed)
        patch_emb = self.get_image_feature(feats[:, 1])
        word_emb, sents = self.get_text_feature(progression_text)
        return patch_emb, word_emb

    def forward(self, samples):
        # for pretraining task
        progression = samples['progression']
        reversed_progression = samples['reversed_progression']
        cur_caption = samples['curr_caption']
        prior_caption = samples['prior_caption']
        image = samples["image"]
        prior_image = samples["prior_image"]

        # 处理图片
        patch_embed = self.visual_encoder(image)  # B x L x D
        prior_patch_embed = self.visual_encoder(prior_image)
        # B x L x D 

        feats = self.STG(prior_patch_embed, patch_embed)
        #reversed_feats = self.STG(patch_embed, prior_patch_embed)
        # B x T x L x D

        prior_patch_feat, patch_feat, cur_patch_feat = feats[:, 0], feats[:, 1], feats[:, 2]
        #reversed_patch_feat = reversed_feats[:, 1]

        patch_emb = self.get_image_feature(patch_feat)
        #reversed_patch_emb = self.get_image_feature(reversed_patch_feat)
        prior_patch_emb = self.get_image_feature(prior_patch_feat)
        cur_patch_emb = self.get_image_feature(cur_patch_feat)

        # 处理文本
        word_emb, sents = self.get_text_feature(progression)
        #reversed_word_emb, reversed_sents = self.get_text_feature(reversed_progression)
        cur_word_emb, cur_sents = self.get_text_feature(cur_caption)
        prior_word_emb, prior_sents = self.get_text_feature(prior_caption)

        # 获取动量特征
        with torch.no_grad():
            self._momentum_update()
            feats_m = self.STG_m(prior_patch_embed, patch_embed)
            #reversed_feats_m = self.STG_m(patch_embed, prior_patch_embed)

            patch_emb_m = self.get_image_feature_m(feats_m[:, 1])
            #reversed_patch_emb_m = self.get_image_feature_m(reversed_feats_m[:, 1])
            prior_patch_emb_m = self.get_image_feature_m(feats_m[:, 0])
            cur_patch_emb_m = self.get_image_feature_m(feats_m[:, 2])

            # text
            word_emb_m, _ = self.get_text_feature_m(progression)
            #reversed_word_emb_m, _ = self.get_text_feature_m(reversed_progression)
            cur_word_emb_m, _ = self.get_text_feature_m(cur_caption)
            prior_word_emb_m, _ = self.get_text_feature_m(prior_caption)

            static_sim_i2t_m, static_sim_t2i_m = self.get_token_similarity(
                cur_patch_emb_m, prior_patch_emb_m, cur_word_emb_m, prior_word_emb_m, cur_sents, prior_sents)
            #progress_sim_i2t_m, progress_sim_t2i_m = self.get_token_similarity(
            #    patch_emb_m, reversed_patch_emb_m, word_emb_m, reversed_word_emb_m, sents, reversed_sents)
            progress_sim_i2t_m, progress_sim_t2i_m = self.get_single_token_similarity(
                patch_emb_m, word_emb_m, sents)
            
            static_sim_targets = torch.zeros(static_sim_i2t_m.size()).to(feats.device)
            static_sim_targets.fill_diagonal_(1) 
            progress_sim_targets = torch.zeros(progress_sim_i2t_m.size()).to(feats.device)
            progress_sim_targets.fill_diagonal_(1)       
            alpha = 0.2
            static_sim_i2t_targets = alpha * F.softmax(static_sim_i2t_m, dim=1) + (1 - alpha) * static_sim_targets
            static_sim_t2i_targets = alpha * F.softmax(static_sim_t2i_m, dim=1) + (1 - alpha) * static_sim_targets  
            progress_sim_i2t_targets = alpha * F.softmax(progress_sim_i2t_m, dim=1) + (1 - alpha) * progress_sim_targets
            progress_sim_t2i_targets = alpha * F.softmax(progress_sim_t2i_m, dim=1) + (1 - alpha) * progress_sim_targets      

        static_sim_i2t, static_sim_t2i = self.get_token_similarity(
                cur_patch_emb, prior_patch_emb, cur_word_emb, prior_word_emb, cur_sents, prior_sents)
        #progress_sim_i2t, progress_sim_t2i = self.get_token_similarity(
        #        patch_emb, reversed_patch_emb, word_emb, reversed_word_emb, sents, reversed_sents)
        progress_sim_i2t, progress_sim_t2i = self.get_single_token_similarity(
                patch_emb, word_emb, sents)
        static_loss = -torch.sum(F.log_softmax(static_sim_i2t, dim=1)*static_sim_i2t_targets,dim=1).mean() - torch.sum(F.log_softmax(static_sim_t2i, dim=1)*static_sim_t2i_targets,dim=1).mean() 
        progress_loss = -torch.sum(F.log_softmax(progress_sim_i2t, dim=1)*progress_sim_i2t_targets,dim=1).mean() - torch.sum(F.log_softmax(progress_sim_t2i, dim=1)*progress_sim_t2i_targets,dim=1).mean() 

        # 重建loss
        rec_cur_patch_feat = self.proj_progress(prior_patch_feat.detach() + patch_feat)
        #rec_prior_patch_feat = self.proj_reverse(cur_patch_feat.detach() + reversed_patch_feat)

        rec_loss = self.rec_loss(cur_patch_feat.detach(), rec_cur_patch_feat) #+ \
        #self.rec_loss(prior_patch_feat.detach(), rec_prior_patch_feat)
        loss = (static_loss + progress_loss + 0.5 * rec_loss) / 3.

        return {"loss": loss, 
                'static_loss': static_loss,
                'progress_loss': progress_loss,
                'rec_loss': rec_loss}

    def forward_temporal_feat(self, image, prior_image):
        patch_embed = self.visual_encoder(image)  # B x L x D
        prior_patch_embed = self.visual_encoder(prior_image)
        # B x L x D 
        feats = self.STG(prior_patch_embed, patch_embed)
        # B x T x L x D

        prior_feat, patch_feat, cur_feat = feats[:, 0], feats[:, 1], feats[:, 2]
        prior_patch_emb = self.get_image_feature(prior_feat)
        patch_emb = self.get_image_feature(patch_feat)
        cur_patch_emb = self.get_image_feature(cur_feat)

        return prior_patch_emb, patch_emb, cur_patch_emb
    
    def forward_atten_feat(self, image, prior_image, layer_idx):
        patch_embed = self.visual_encoder(image)  # B x L x D
        prior_patch_embed = self.visual_encoder(prior_image)
        # B x L x D 
        x, spatial_attn, temporal_attn = self.STG.forward_with_attn(prior_patch_embed, patch_embed, layer_idx)

        return spatial_attn, temporal_attn
    
    def forward_similarity(self, cur_image, prior_image, templ):
        patch_embed = self.visual_encoder(cur_image)  # B x L x D
        prior_patch_embed = self.visual_encoder(prior_image)

        feats = self.STG(prior_patch_embed, patch_embed)
        patch_emb = self.get_image_feature(feats[:, 1])
        
        word_emb_prompt, valid_prompt = [], []
        for i in range(len(templ)):
            word_emb, sents = self.get_text_feature(templ[i])
            pad_mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(word_emb) 
            valid = 1 - pad_mask
            word_emb_prompt.append(word_emb)  # B x N x D
            valid_prompt.append(valid)
        word_emb_prompt = torch.stack(word_emb_prompt, dim=1)  # B x P x N x D

        patch_score = patch_emb.mean(dim=1) @ word_emb_prompt.squeeze().mean(dim=1).t() / self.temperature  # B x P

        #valid = torch.stack(valid_prompt, dim=1)  # B x P x N
        #sim = torch.einsum("bld,bpnd->bpln", patch_emb, word_emb_prompt)  # [B,P,L,N]
        #sim = sim * valid.unsqueeze(2)  # [B,P,L,N]

        #patch_score = sim.max(dim=-1).values              # [B,P,L]
        #patch_score = patch_score.mean(dim=-1) / self.temperature
    
        return patch_score
    

    def save_checkpoint(self, train_loss: float):
        """在预训练阶段，根据训练集 loss 保存当前最优模型"""
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step

        state_dict = self.state_dict()

        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
            "train_loss": float(train_loss),
        }

        ckpt_dir = os.path.join(self.hparams.savedmodel_path, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        save_to = os.path.join(
            ckpt_dir,
            "checkpoint_epoch{}.pth".format(
                current_epoch
            ),
        )
        self.print(f"Saving checkpoint at step {global_step} to {save_to}.")
        torch.save(save_obj, save_to)
    
    def training_step(self, batch, batch_idx):
        result = self(batch)          # {"loss": loss}
        loss = result["loss"]

        self.log_dict(result, prog_bar=True)  

        loss_scalar = float(loss.detach().cpu().item())
        if loss_scalar < self.best_train_loss:
            self.best_train_loss = loss_scalar
            self.print(f"[Pretrain] New best train loss: {loss_scalar:.6f}, saving checkpoint...")
            self.save_checkpoint(loss_scalar)

        return result
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
        