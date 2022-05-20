import os
import random

import numpy as np

import torch
import torch.nn as nn
from torchvision.io import write_video
from torchvision import utils


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True)
        return x / maxes


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg,
                     path,
                     nrow=int(n_sample**0.5),
                     normalize=True,
                     range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def clip_similarity(model, tokenizer, image, description):
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()

    if image.shape[2] != input_resolution:
        image = F.interpolate(image, (input_resolution, input_resolution))
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    image_input = (image - image_mean[:, None, None]) / image_std[:, None,
                                                                  None]
    text_input = tokenizer.tokenize(
        description,
        context_length,
        truncate_text=True,
    ).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (text_features.cpu().numpy() *
                  image_features.cpu().numpy()).sum(1)
    return similarity


def exists(val):
    return val is not None


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def sample_data(loader, sampler=None):
    epoch = -1
    while True:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch


"""
Copied from VQGAN main.py
"""
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
