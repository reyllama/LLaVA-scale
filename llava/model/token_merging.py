import math
import torch
from torch import nn
from typing import Tuple, Callable

# code modified from https://github.com/dbolya/tomesd

class LocalTokenMerging:
    def __init__(self, w: int, h: int, sx: int = 2, sy: int = 2, r: float = 0.5):
        self.w, self.h = w, h
        self.sx, self.sy = sx, sy
        self.r = 0.0
        self.max_r = r
        self.tik = 0.0

        if torch.distributed.get_rank() == 0:
            print(f"[ToMe] Initializing Local Token Merging with w: {w}, h: {h}, sx: {sx}, sy: {sy}, r: {r}")

    def adjust_r(self):
        # self.r = self.max_r * (1 - math.cos((self.tik / 4000) * math.pi / 2)) if self.tik < 4000 else self.max_r
        self.r = self.max_r

    def __call__(self, x: torch.Tensor):
        B, N, C = x.shape
        r = int(self.r * N)

        self.tik += 1
        self.adjust_r()

        if self.tik % 100 == 0:
            print(f"** ToMe info: self.r: {self.r}, self.max_r: {self.max_r}, self.tik: {self.tik}, N: {N}, r: {r}")

        if r <= 0:
            return x

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

        with torch.no_grad():
            hsy, wsx = self.h // self.sy, self.w // self.sx

            rand_idx = torch.randint(self.sy*self.sx, size=(hsy, wsx, 1)).to(x.device)
            
            idx_buffer_view = torch.zeros(hsy, wsx, self.sy*self.sx, device=x.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
            idx_buffer_view = idx_buffer_view.view(hsy, wsx, self.sy, self.sx).transpose(1, 2).reshape(hsy * self.sy, wsx * self.sx)

            if (hsy * self.sy) < self.h or (wsx * self.sx) < self.w:
                idx_buffer = torch.zeros(self.h, self.w, device=x.device, dtype=torch.int64)
                idx_buffer[:(hsy * self.sy), :(wsx * self.sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

            del idx_buffer, idx_buffer_view

            num_dst = hsy * wsx
            a_idx = rand_idx[:, num_dst:, :]
            b_idx = rand_idx[:, :num_dst, :]

            def split(x):
                C = x.shape[-1]
                src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            metric = x / x.norm(dim=-1, keepdim=True)
            a, b = split(metric)
            scores = a @ b.transpose(-1, -2)

            r = min(a.shape[1], r)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            # dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
            # if torch.distributed.get_rank() == 0:
            #     print(f"** [ToMe] r: {r}")
            #     print(f"** [ToMe] unm_idx.shape: {unm_idx.shape}, src_idx.shape: {src_idx.shape}, dst_idx.shape: {dst_idx.shape}")
            #     print(f"** [ToMe] unm_idx: {unm_idx[0,:,0]}, src_idx: {src_idx[0,:,0]}, dst_idx: {dst_idx[0,:,0]}")


            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            for b in range(B):
                mask[b, src_idx[b, :, 0]] = False
            out = x[mask].view(B, N - r, -1)

            return out

            # n, t1, c = a.shape

            # unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            # src = gather(a, dim=-2, index=src_idx.expand(n, r, c))
            # dst = b.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="mean")

            # original_indices = torch.arange(N, device=x.device).reshape(1, N, 1).expand(B, N, 1)
            # unm_indices = gather(original_indices, dim=1, index=unm_idx.expand(B, t1 - r, 1))
            # dst_indices = gather(original_indices, dim=1, index=b_idx.expand(B, num_dst, 1))
            # if torch.distributed.get_rank() == 0:
            #     print(f"** [ToMe] unm_indices.shape: {unm_indices.shape}, dst_indices.shape: {dst_indices.shape}")
            #     print(f"** [ToMe] unm_indices: {unm_indices[0,:,0]}, dst_indices: {dst_indices[0,:,0]}")

            # new_indices = torch.cat([unm_indices, dst_indices], dim=1)
            # new_tokens = torch.cat([unm, dst], dim=1)
            # if torch.distributed.get_rank() == 0:
            #     print(f"** [ToMe] new_indices.shape: {new_indices.shape}, new_tokens.shape: {new_tokens.shape}")
            #     print(f"** [ToMe] new_indices: {new_indices[0,:,0]}")

            # # Sort by the original indices to maintain relative order
            # sorted_indices = new_indices.argsort(dim=1)
            # if torch.distributed.get_rank() == 0:
            #     print(f"** [ToMe] sorted_indices.shape: {sorted_indices.shape}")
            #     print(f"** [ToMe] sorted_indices: {sorted_indices[0,:,0]}")
            # result = gather(new_tokens, dim=1, index=sorted_indices.expand(B, N - r, C))

        return result

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)