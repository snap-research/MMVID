import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from mmvid_pytorch.modules import AxialPositionalEmbeddingList
from utils.utils import DivideMax

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# main classes

# helper


def define_transformer():
    return None


def warp_video_with_color(video):
    # video (n, t, 3, h, w)
    out = []
    for n in range(video.shape[0]):
        x = video[n]  # x (c, h, w)
        c_shift = torch.rand(1) - 0.5
        c_shift = c_shift.to(x.device)
        m = torch.zeros_like(x)
        num = random.randint(0, 3)
        if num == 0:
            m.data += c_shift
        elif num == 1:
            m[:, 0].data += c_shift
        elif num == 2:
            m[:, 1].data += c_shift
        else:
            m[:, 2].data += c_shift
        out.append(torch.clamp(x + m, 0, 1))
    return torch.stack(out)


# main DALL-E class


class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        cvae=None,
        num_text_tokens=10000,
        text_seq_len=256,
        loss_img_weight=7,
        stable=False,
        which_transformer='none',
        num_visuals=1,
        num_targets=1,
        **kwargs,
    ):
        super().__init__()
        # assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024)), 'vae must be an instance of DiscreteVAE'

        assert num_visuals > 0
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2**vae.num_layers))
        image_seq_len = image_fmap_size**2
        target_seq_len = image_seq_len * num_targets
        visual_seq_len = image_seq_len * num_visuals
        control_seq_len = text_seq_len + visual_seq_len
        self.insert_sep = False

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)
        num_visual_tokens = num_image_tokens + visual_seq_len  # reserve unique padding tokens for each position
        num_control_tokens = num_text_tokens + num_visual_tokens

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim)  # +1 for <bos>
        if num_targets == 1:
            self.image_pos_emb = AxialPositionalEmbedding(
                dim, axial_shape=(image_fmap_size, image_fmap_size))
        else:
            self.image_pos_emb = AxialPositionalEmbedding(
                dim,
                axial_shape=(num_targets, image_fmap_size, image_fmap_size))

        if num_visuals > 0:
            # self.visual_emb = nn.Embedding(num_image_tokens + 2, dim)  # TODO: for masking+separate visual
            self.visual_emb = nn.Embedding(num_visual_tokens, dim)
            self.visual_pos_emb = AxialPositionalEmbeddingList(
                dim,
                num_visuals,
                axial_shape=(image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens  # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens
        self.num_visual_tokens = num_visual_tokens
        self.num_control_tokens = num_control_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.target_seq_len = target_seq_len
        self.visual_seq_len = visual_seq_len
        self.control_seq_len = control_seq_len
        self.num_visuals = num_visuals
        self.num_targets = num_targets
        self.image_fmap_size = image_fmap_size

        self.special_token_lut = {
            '[REL]': 0,
            '[ST1]': 1,
            '[ST2]': 2,
            '[ST3]': 3,
        }
        self.num_special_tokens = len(self.special_token_lut)
        self.num_estimation_tokens = 2  # rel, fdl
        self.special_emb = nn.Embedding(self.num_special_tokens, dim)
        self.estimation_pos_emb = nn.Embedding(self.num_estimation_tokens, dim)

        seq_len = text_seq_len + target_seq_len + visual_seq_len
        if num_visuals > 0:
            total_tokens = num_text_tokens + num_image_tokens + num_visual_tokens
        else:
            total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae
        self.cvae = cvae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained
        set_requires_grad(self.cvae, False)  # freeze VAE from being trained

        self.which_transformer = which_transformer
        if which_transformer.startswith('openai_clip'):
            from mmvid_pytorch.transformers.clip_model import OpenAICLIPTransformer
            self.transformer = OpenAICLIPTransformer(
                seq_len,
                which_transformer,
                model_path=kwargs['openai_clip_path'],
            )
        else:
            raise NotImplementedError

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        if num_visuals > 0:
            logits_mask = (torch.block_diag(
                torch.ones(text_seq_len, num_text_tokens),
                torch.ones(visual_seq_len, num_visual_tokens),
                torch.ones(target_seq_len, num_image_tokens),
            ) == 0).unsqueeze(0)
        else:
            logits_mask = (torch.block_diag(
                torch.ones(text_seq_len, num_text_tokens),
                torch.ones(target_seq_len, num_image_tokens),
            ) == 0).unsqueeze(0)

        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_vis_weight = 1.
        self.loss_img_weight = loss_img_weight

        self.eraser = T.RandomErasing(p=1,
                                      scale=(0.4, 0.8),
                                      ratio=(0.5, 2),
                                      value=-1)  # erase visual

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip=None,
        visual=None,
        mask=None,
        filter_thres=0.5,
        temperature=1.,
        erase_visual=False,
        vc_mode=None,
        face_mode=None,
        **kwargs,
    ):
        vae, text_seq_len, target_seq_len, num_control_tokens = self.vae, self.text_seq_len, self.target_seq_len, self.num_control_tokens
        total_len = text_seq_len + target_seq_len

        text = text[:, :text_seq_len]  # make sure text is within bounds
        out = text

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(
                text,
                visual=visual,
                target=image,
                mask=mask,
                vc_mode=vc_mode,
                face_mode=face_mode,
                erase_visual=erase_visual,
                erase_visual_half=True,
            )[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            sample -= (
                num_control_tokens if is_image else 0
            )  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value=True)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -target_seq_len:]
        if self.num_targets > 1:
            img_seq = rearrange(img_seq,
                                'b (t n) -> (b t) n',
                                n=self.image_seq_len)
            images = vae.decode(img_seq)
            images = rearrange(images,
                               '(b t) c h w -> b t c h w',
                               t=self.num_targets)
        else:
            images = vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images, [], None

    def get_image_tokens(self,
                         image,
                         reshape=True,
                         insert_sep=False,
                         which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        if isinstance(image, list):
            assert len(image[0].shape
                       ) == 4, 'image should be list of 4d image tensors'
            image = torch.stack(image, dim=1)
        if len(image.shape) == 4:
            image = image.unsqueeze(1)
        is_raw_image = len(image.shape) == 5  # [B, T, C, H, W]
        if is_raw_image:
            b, t, c, h, w = image.shape
            image_size = vae.image_size
            assert (c, h, w) == (
                3, image_size, image_size
            ), f'invalid image of dimensions {image.shape} passed in during training'
            image = rearrange(image, 'b t c h w -> (b t) c h w')
            image = vae.get_codebook_indices(image)  # ((b t) n)
            if reshape:
                if insert_sep:
                    image = rearrange(image, '(b t) n -> b t n', t=t)
                    image = torch.cat(
                        (image, torch.empty(
                            b, t, 1, device=image.device).long().fill_(
                                self.image_token_lut['[SEP]'])),
                        dim=2)
                    image = rearrange(image, 'b t n -> b (t n)')
                else:
                    image = rearrange(image, '(b t) n -> b (t n)', t=t)
        return image

    @torch.no_grad()
    def recon_images(self, images, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images,
                                        reshape=False,
                                        which_vae=which_vae)
        images = vae.decode(img_seq)
        # images = rearrange(images, '(b t) c h w -> b t c h w', t = self.num_targets)
        return images

    def random_erase_codebook(self, image, eraser, erase_half=False):
        mask_val = -1
        image = rearrange(image,
                          'b (t h w) -> b t h w',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        if erase_half:
            image_ = image
            image_[:, :, self.image_fmap_size // 2:, :] = mask_val
        else:
            image_ = torch.stack([eraser(c) for c in image], dim=0)
        image = rearrange(image_,
                          'b t h w -> b (t h w)',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        return image

    def erase_codebook_face(self, image, vc_mode, face_mode=None):
        mask_val = -1
        image = rearrange(image,
                          'b (t h w) -> b t h w',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        if vc_mode == 'face_8x8':
            image_ = mask_val + torch.zeros_like(image).long()
            if face_mode is None:
                face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
            if face_mode == 'eyes_nose':  # eyes and nose
                image_[:, :, 2:5, 1:7] = image[:, :, 2:5, 1:7]
            else:  # mouth
                image_[:, :, 5:7, 2:6] = image[:, :, 5:7, 2:6]
            image = image_
        elif vc_mode == 'face2_8x8':
            image_ = mask_val + torch.zeros_like(image).long()
            image_[:, 0, ...] = image[:, 0, ...]
            image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            image = image_
        elif vc_mode == 'mask_8x8' or vc_mode == 'mask2_8x8':
            if face_mode is None:
                which_strategy = np.random.choice([1, 2, 3],
                                                  p=[0.5, 0.25, 0.25])
            else:
                which_strategy = 3
            if which_strategy == 1:
                image_ = image
            elif which_strategy == 2:
                image_ = mask_val + torch.zeros_like(image).long()
                image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            elif which_strategy == 3:
                image_ = mask_val + torch.zeros_like(image).long()
                image_[:, :, 1:7, 1:7] = image[:, :, 1:7, 1:7]
                image = image_
        elif vc_mode == 'shape_4x4':
            image[:, :, 1:3, 1:3] = mask_val
        else:
            raise NotImplementedError
        image = rearrange(image,
                          'b t h w -> b (t h w)',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        return image

    def forward(
        self,
        text,
        visual=None,
        target=None,
        return_loss=False,
        erase_visual=False,
        erase_visual_half=False,
        vc_mode=None,
        face_mode=None,
        visual_aug_mode=None,
        **kwargs,
    ):
        assert text.shape[
            -1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        device, total_seq_len = text.device, self.total_seq_len
        batch_size = text.shape[0]
        image = target

        text_augment = False

        # make sure padding in text tokens get unique padding token id

        text_range = torch.arange(self.text_seq_len, device=device) + (
            self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>
        bos = 0
        text = F.pad(text, (1, 0), value=bos)  # TODO: !!!

        text_emb = self.text_emb(text)
        if text_augment == 'noise':
            text_emb += 0.1 * torch.randn_like(text_emb)
        text_emb += self.text_pos_emb(
            torch.arange(text.shape[1], device=device))
        tokens = text_emb
        seq_len = text.shape[1]

        # visual tokens

        if exists(visual) and not is_empty(visual):
            if visual_aug_mode == 'motion_color' and random.random() < 0.9:
                visual_ = visual.detach().clone()
                visual_[:, 1:, ...] = warp_video_with_color(visual[:, 1:, ...])
                visual = visual_
            visual = self.get_image_tokens(visual,
                                           insert_sep=self.insert_sep,
                                           which_vae='cvae')
            if erase_visual:
                visual = self.random_erase_codebook(visual, self.eraser,
                                                    erase_visual_half)
            if vc_mode is not None:
                visual = self.erase_codebook_face(visual, vc_mode, face_mode)
        else:
            visual = -torch.ones(
                batch_size, self.visual_seq_len, device=device).long()
        visual_range = torch.arange(self.visual_seq_len, device=device) + (
            self.num_visual_tokens - self.visual_seq_len)
        visual = torch.where(visual == -1, visual_range, visual)
        visual_emb = self.visual_emb(visual)
        visual_pos_emb = self.visual_pos_emb(visual_emb)
        visual_emb += visual_pos_emb
        tokens = torch.cat((tokens, visual_emb), dim=1)
        seq_len += visual.shape[1]

        if exists(image) and not is_empty(image):
            image = self.get_image_tokens(image)
            image_emb = self.image_emb(image)
            image_pos_emb = self.image_pos_emb(image_emb)
            image_emb += image_pos_emb

            tokens = torch.cat((tokens, image_emb), dim=1)
            seq_len += image.shape[1]

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max

        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'

        offsetted_visual = visual + self.num_text_tokens
        offsetted_image = image + self.num_control_tokens
        labels = torch.cat((text[:, 1:], offsetted_visual, offsetted_image),
                           dim=1)  # text[:,0] is <bos>

        logits = rearrange(logits, 'b n c -> b c n')

        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len],
                                    labels[:, :self.text_seq_len])
        if self.num_visuals > 0:
            loss_vis = F.cross_entropy(
                logits[:, :, self.text_seq_len:self.control_seq_len],
                labels[:, self.text_seq_len:self.control_seq_len])
        else:
            loss_vis = torch.tensor(0.0, device=device)
        loss_img = F.cross_entropy(logits[:, :, self.control_seq_len:],
                                   labels[:, self.control_seq_len:])

        loss = (loss_text + self.loss_vis_weight * loss_vis +
                self.loss_img_weight * loss_img) / (self.loss_img_weight +
                                                    self.loss_vis_weight + 1)

        zero = torch.tensor(0.0, device=device)
        return loss, zero, zero
