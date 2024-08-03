import torch
from torch import nn
from typing import Tuple, Callable

# code reliant on https://github.com/dbolya/tomesd

class SDlikeTokenMerging:
    def __init__(self, w: int, h: int, sx: int = 2, sy: int = 2, r: float = 0.5):
        self.w, self.h = w, h
        self.sx, self.sy = sx, sy
        self.r = 0.0
        self.max_r = r
        self.tik = 0

    def adjust_r(self):
        self.r = min(self.max_r, self.tik / 4000 * self.max_r)
        
    def __call__(self, x: torch.Tensor):
        B, N, _ = x.shape
        r = int(self.r * N)

        if r <= 0:
            return x

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather
        
        with torch.no_grad():
            hsy, wsx = self.h // self.sy, self.w // self.sx

            # For each sy by sx kernel, randomly assign one token to be dst and the rest src
            rand_idx = torch.randint(self.sy*self.sx, size=(hsy, wsx, 1)).to(x.device)
            
            # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
            idx_buffer_view = torch.zeros(hsy, wsx, self.sy*self.sx, device=x.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
            idx_buffer_view = idx_buffer_view.view(hsy, wsx, self.sy, self.sx).transpose(1, 2).reshape(hsy * self.sy, wsx * self.sx)

            # Image is not divisible by sx or sy so we need to move it into a new buffer
            if (hsy * self.sy) < self.h or (wsx * self.sx) < self.w:
                idx_buffer = torch.zeros(self.h, self.w, device=x.device, dtype=torch.int64)
                idx_buffer[:(hsy * self.sy), :(wsx * self.sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

            # We're finished with these
            del idx_buffer, idx_buffer_view

            # rand_idx is currently dst|src, so split them
            num_dst = hsy * wsx
            a_idx = rand_idx[:, num_dst:, :] # src
            b_idx = rand_idx[:, :num_dst, :] # dst

            def split(x):
                C = x.shape[-1]
                src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            # Cosine similarity between A and B
            metric = x / x.norm(dim=-1, keepdim=True)
            a, b = split(metric)
            scores = a @ b.transpose(-1, -2)

            # Can't reduce more than the # tokens in src
            r = min(a.shape[1], r)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

            n, t1, c = a.shape

            unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = gather(a, dim=-2, index=src_idx.expand(n, r, c))
            dst = b.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="mean")

        self.tik += 1
        if self.tik > 1000:
            self.adjust_r()

        return torch.cat([unm, dst], dim=1)


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)