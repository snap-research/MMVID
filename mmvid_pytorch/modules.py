import numpy as np
import torch
from torch import nn
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange


class AxialPositionalEmbeddingList(nn.Module):
    def __init__(
            self,
            dim=512,
            num=None,
            axial_shape=(),
    ):
        super().__init__()
        if num is None:
            num = axial_shape[0]
            axial_shape = axial_shape[1:]
        self.dim = dim
        self.num = num
        self.axial_shape = axial_shape
        self.chunk_size = np.prod(axial_shape)
        self.seq_len = num * self.chunk_size
        self.module_list = nn.ModuleList([
            AxialPositionalEmbedding(dim, axial_shape=axial_shape)
            for _ in range(num)
        ])

    def forward(self, emb):
        # emb: b (t n) d
        if emb.shape[1] > self.seq_len:  # [SEP] inserted
            chunks = torch.chunk(emb, self.num, dim=1)
            pos_emb = torch.stack([
                module(chunk[:, :-1])
                for chunk, module in zip(chunks, self.module_list)
            ],
                                  dim=1)
            pos_emb = torch.cat((pos_emb,
                                 torch.zeros(emb.shape[0],
                                             len(chunks),
                                             1,
                                             emb.shape[-1],
                                             device=emb.device).long()),
                                dim=2)
            pos_emb = rearrange(pos_emb, 'b t n d -> b (t n) d')
        else:
            chunks = torch.chunk(emb, self.num, dim=1)
            pos_emb = torch.cat([
                module(chunk)
                for chunk, module in zip(chunks, self.module_list)
            ],
                                dim=1)
        return pos_emb
