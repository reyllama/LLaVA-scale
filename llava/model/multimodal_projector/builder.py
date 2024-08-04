import torch
import torch.nn as nn
import re
from ..multimodal_encoder.clip_encoder import CLIPTextTower

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def split_tensor(tensor, n, w):

    '''
        Reshape (b,m,d) tensor into (b,n,w,d) tensor (with overlapping windows) 
    '''

    tensor_length = tensor.size(1)
    subarrays = []
    
    step = max(1, (tensor_length - w) // (n - 1))
    
    for i in range(n):
        start_index = i * step
        end_index = start_index + w
        if end_index > tensor_length:
            start_index = tensor_length - w
            end_index = tensor_length
        subarray = tensor[:, start_index:end_index]
        subarrays.append(subarray)
    
    out_tensor = torch.stack(subarrays).transpose(0, 1)

    return out_tensor

class SwinQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.instruction_tower = CLIPTextTower(getattr(config, 'instruction_tower', 'openai/clip-vit-large-patch14'), 
                                          config)
        self.instruction_tower.load_model()
        self.inst_projection = nn.Linear(config.instruction_hidden_size, config.hidden_size)
        self.Queries = nn.Parameter(torch.randn(1, config.num_queries, config.hidden_size)) # TODO
        self.norm = nn.LayerNorm(config.hidden_size)
        self.to_kv_inst = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.to_out_inst = nn.Linear(config.hidden_size, config.hidden_size)
        self.to_kv_mmodal = nn.Linear(config.mmodal_hidden_size, config.hidden_size * 2)
        self.to_out_mmodal = nn.Linear(config.hidden_size, config.hidden_size)
        self.window_size = getattr(config, 'window_size', 512)

    def apply_attention(self, q, k, v):

        attn = torch.einsum('b n d, b m d -> b n m', q, k)
        attn = attn / (k.shape[-1] ** 0.5)
        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b n m, b m d -> b n d', attn, v)

        return out

    def apply_swin_attention(self, q, k, v):
        
        b, n, d = q.shape
        k = split_tensor(k, n, self.window_size)
        v = split_tensor(v, n, self.window_size)

        attn = torch.einsum('b n d, b n w d -> b n w', q, k)
        attn = attn / (k.shape[-1] ** 0.5)
        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b n w, b n w d -> b n d', attn, v)
        
        return out

    def forward(self, x_mmodal: torch.Tensor, x_inst: torch.Tensor):

        # print(f"x_mmodal: {x_mmodal.shape}") # (batch, 256, 1024)

        inst_feats = self.instruction_tower(x_inst)
        b, n, _ = inst_feats.shape
        # print(f"inst_feats: {inst_feats.shape}") # (batch, n, 768)
        inst_feats = self.inst_projection(inst_feats)
        k, v = self.to_kv_inst(inst_feats).chunk(2, dim=-1)
        k, v = self.norm(k), self.norm(v)
        q = self.Queries.expand(b, -1, -1)
        q = self.apply_attention(q, k, v)
        q = self.to_out_inst(q)
        q = self.norm(q)
        k, v = self.to_kv_mmodal(x_mmodal).chunk(2, dim=-1)
        k, v = self.norm(k), self.norm(v)
        out = self.apply_swin_attention(q, k, v)
        out = self.to_out_mmodal(out)
        return self.norm(out)

def build_projector(config, projector_name_type=None, delay_load=False, **kwargs):
    projector_type = getattr(config, projector_name_type, 'linear') # linear

    if projector_type == 'linear':
        if projector_name_type == 'mm_projector_type':
            return nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif projector_name_type == "audio_projector_type":
            return nn.Linear(config.audio_hidden_size, config.hidden_size)
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')

    elif projector_type == 'swinqformer':
        swin_config = config
        swin_config.instruction_hidden_size = getattr(config, 'instruction_hidden_size', 768)
        swin_config.hidden_size = getattr(config, 'hidden_size', 1024)
        swin_config.num_queries = getattr(config, 'num_queries', 128)
        swin_config.mmodal_hidden_size = getattr(config, 'mmodal_hidden_size', 1024)
        return SwinQFormer(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

