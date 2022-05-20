from pathlib import Path
from math import sqrt
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

import torch
from torch import nn

from einops import rearrange

# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841


class VQGanVAE1024(nn.Module):
    def __init__(self, vae_path=None, image_size=None):
        super().__init__()

        model_filename = 'vqgan.1024.model.ckpt'
        config_filename = 'vqgan.1024.config.yml'

        config_path = str(Path('mmvid_pytorch') / 'data' / config_filename)
        config = OmegaConf.load(config_path)
        if image_size:
            config.model.params['ddconfig']['resolution'] = image_size
        model = VQModel(**config.model.params)

        if vae_path is not None:
            state = torch.load(vae_path, map_location='cpu')['state_dict']
            model.load_state_dict(state, strict=False)

        self.model = model

        self.num_layers = 4
        self.image_size = 256
        self.num_tokens = 1024

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices.squeeze(), '(b n) -> b n', b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        # one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        # z = (one_hot_indices @ self.model.quantize.embedding.weight)
        # NOTE: original is the above, should be equivalent to the below
        z = self.model.quantize.embedding(img_seq)

        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def decode_train(self, probs):
        # probs [B, N, D]
        b, n, d = probs.shape
        one_hot_indices = probs

        z = (one_hot_indices @ self.model.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented
