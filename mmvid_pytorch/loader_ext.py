from pathlib import Path
from random import randint, choice
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pickle
import os
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import torchvision.transforms.functional as TF

import decord

decord.bridge.set_bridge("torch")

import mm_vox_celeb.pcfg as pcfg

ATTR = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]
ATTR_NP = np.array(ATTR)
NAME = [
    a.replace('No_', '').replace('Wearing_', '').replace('_', ' ').lower()
    for a in ATTR
]
NAME[0] = '5 o\'clock shadow'
NAME = np.array(NAME)

GET_NAME = {a: NAME[i] for i, a in enumerate(ATTR)}

ATTR_VERB = {
    '5_o_Clock_Shadow': 'has',
    'Arched_Eyebrows': 'has',
    'Attractive': 'is',
    'Bags_Under_Eyes': 'has',
    'Bald': 'is',
    'Bangs': 'has',
    'Big_Lips': 'has',
    'Big_Nose': 'has',
    'Black_Hair': 'has',
    'Blond_Hair': 'has',
    'Blurry': 'is',
    'Brown_Hair': 'has',
    'Bushy_Eyebrows': 'has',
    'Chubby': 'is',
    'Double_Chin': 'has',
    'Eyeglasses': 'wear',
    'Goatee': 'wear',
    'Gray_Hair': 'has',
    'Heavy_Makeup': 'has',
    'High_Cheekbones': 'has',
    'Male': 'is',
    'Mouth_Slightly_Open': 'na',
    'Mustache': 'has',
    'Narrow_Eyes': 'has',
    'No_Beard': 'has',
    'Oval_Face': 'has',
    'Pale_Skin': 'has',
    'Pointy_Nose': 'has',
    'Receding_Hairline': 'has',
    'Rosy_Cheeks': 'has',
    'Sideburns': 'has',
    'Smiling': 'is',
    'Straight_Hair': 'has',
    'Wavy_Hair': 'has',
    'Wearing_Earrings': 'wear',
    'Wearing_Hat': 'wear',
    'Wearing_Lipstick': 'wear',
    'Wearing_Necklace': 'wear',
    'Wearing_Necktie': 'wear',
    'Young': 'is',
}

NEGATE_IDX = [ATTR.index(a) for a in ATTR if a.startswith('No_')]
GENDER_IDX = ATTR.index('Male')

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
VID_EXTENSIONS = ['.mp4', '.avi']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def is_video_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in VID_EXTENSIONS)


def to_tensor(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    try:
        image_tensor = TF.to_tensor(image)
    except:
        return None
    return image_tensor


def read_frames_imagestack(video_path, frame_idxs=None):
    imgs = Image.open(video_path).convert('RGB')  # size (W, H)
    imgs = np.array(imgs)  # shape (H, W, C)
    horizontal = imgs.shape[1] > imgs.shape[0]
    shorter, longer = min(imgs.shape[0],
                          imgs.shape[1]), max(imgs.shape[0], imgs.shape[1])
    vlen = longer // shorter
    frames = np.stack(np.split(imgs, vlen, axis=1 if horizontal else 0))
    if frame_idxs:
        frames = frames[frame_idxs, ...]
    frames = torch.from_numpy(frames).permute(
        0, 3, 1, 2).float() / 255  # tensor of shape (T, C, H, W), range (0, 1)
    return frames


class VoxDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        cache=None,
        return_vc=False,
        video_only=False,
        keys=None,
        return_neg=False,
        attr_mode='mask+text',
        sample_label=False,
        cat1=[],
        args=None,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.deterministic = deterministic
        self.sample_label = sample_label
        self.args = args

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.min_len = max(min_len, (self.frame_num - 1) * self.frame_step + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_neg = return_neg
        self.video_only = video_only
        self.attr_mode = attr_mode
        self.cat1 = cat1

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(
                    tqdm(os.listdir(video_root), desc="Counting videos")):
                key = video  # no stem
                text = key + '.txt'
                if os.path.isdir(os.path.join(video_root,
                                              key)) and text in text_files:
                    frames = natsorted(
                        os.listdir(os.path.join(video_root, key)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(video_root, key,
                                                  frame_name)):
                        frame_list.append(
                            os.path.join('video', key, frame_name))
                if len(frame_list) > 0:
                    # add entry
                    keys_list.append(key)
                    text_dict[key] = os.path.join('txt', text)
                    video_dict[key] = frame_list
                    length_dict[key] = len(frame_list)
                # clear
                frame_list = frames = None
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            assert len(self.keys) > 0
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        attr_cache = path.parent / (path.name + '_attr_dict_vox2.pkl')
        if attr_cache.exists():
            with open(attr_cache, 'rb') as f:
                self.attr_dict = pickle.load(f)
        else:
            attr_dict = {'pid': {}, 'attr': {}, 'cat1': {}}
            for k in tqdm(self.keys):
                # pid = k.split('#')[0]
                pid = '#'.join(k.split('#')[:2])
                if pid in attr_dict['pid']:
                    attr_dict['pid'][pid].append(k)
                else:
                    attr_dict['pid'][pid] = [k]
                y = self._get_label(k).split(',')
                for j in range(len(y)):
                    if y[j] == '1':
                        if j in attr_dict['cat1']:
                            attr_dict['cat1'][j].append(k)
                        else:
                            attr_dict['cat1'][j] = [k]
            with open(attr_cache, 'wb') as f:
                pickle.dump(attr_dict, f)
            self.attr_dict = attr_dict

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = keys_keep

        attr_dict = {}
        for attr_type in self.attr_dict:
            attr_dict[attr_type] = {}
            for attr in self.attr_dict[attr_type].keys():
                attr_dict[attr_type][attr] = list(
                    set(self.attr_dict[attr_type][attr]) & set(keys_keep))
        self.attr_dict = attr_dict

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        if deterministic:
            self.video_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.video_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_video(self, index):
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = 0 if self.deterministic else random.randint(
            0, video_len - (self.frame_num - 1) * self.frame_step -
            1)  # inclusive
        frames = []
        for i in range(start_idx, start_idx + self.frame_num * self.frame_step,
                       self.frame_step):
            img = Image.open(os.path.join(self.root, self.videos[key][i]))
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.video_transform(frames)
        if True:
            idx = 0 if self.deterministic else random.randint(0, video_len - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.video_transform(to_tensor(visual))
            return frames, key, visual, start_idx
        return frames, key, start_idx

    def _get_video_by_key(self, key):
        video_len = self.lengths[key]
        start_idx = 0 if self.deterministic else random.randint(
            0, video_len - (self.frame_num - 1) * self.frame_step -
            1)  # inclusive
        frames = []
        for i in range(start_idx, start_idx + self.frame_num * self.frame_step,
                       self.frame_step):
            img = Image.open(os.path.join(self.root, self.videos[key][i]))
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.video_transform(frames)
        return frames, start_idx

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        frame = Image.open(os.path.join(self.root,
                                        self.videos[key][frame_idx]))
        frame = self.video_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.video_transform(to_tensor(visual))
            return frame, key, visual
        return frame, key

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        key = self.keys[video_id]
        frame = Image.open(os.path.join(self.root, self.videos[key][frame_id]))
        frame = self.video_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, key

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        key = self.keys[video_id]
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(
                os.path.join(self.root, self.videos[key][frame_id + i]))
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.video_transform(frames)
        return frames, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def _get_label(self, key):
        label_file = Path(
            os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        label = label_file.read_text().rstrip()
        return label

    def _sample_negative_label(self, key):
        label = self._get_label(key)
        key_ = choice(self.keys)
        label_ = self._get_label(key_)
        while label_ == label:
            key_ = choice(self.keys)
            label_ = self._get_label(key_)
        return key_

    def _tokenize_text(self, description):
        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description
        return tokenized_text

    def __getitem__(self, ind):
        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual, start_idx = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.video_only:
            description = 'dummy text'
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description
            return tokenized_text, image_tensor, visual

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        draw_style = 'style1'

        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
            if self.attr_mode == 'text':
                visuals = visual.unsqueeze(0)
            elif self.attr_mode == 'cat1':
                image_tensor = []
                tokenized_text = []
                for yi in self.cat1:
                    indd = ind % len(self.attr_dict['cat1'][yi])
                    k = self.attr_dict['cat1'][yi][indd]
                    frames, _ = self._get_video_by_key(k)
                    image_tensor.append(frames)
                    desc = pcfg.generate_phrase(
                        (True, 1), (ATTR_VERB[ATTR[yi]], NAME[yi]))
                    desc = 'A person' + desc[2:]
                    tokenized_text.append(self._tokenize_text(desc))
                image_tensor = torch.stack(image_tensor)
                tokenized_text = torch.stack(
                    tokenized_text
                ) if self.tokenizer is not None else tokenized_text
                return image_tensor, tokenized_text
            elif self.attr_mode == 'cat2':
                image_tensor = []
                tokenized_text = []

                # gender
                yi = ATTR.index('Male')
                indd = ind
                key = self.keys[indd]
                frames, _ = self._get_video_by_key(key)
                image_tensor.append(frames)
                if self._get_label(key).split(',')[yi] == '1':
                    desc = 'A boy.' if ind % 2 == 0 else 'A guy.'
                else:
                    desc = 'A girl.' if ind % 2 == 0 else 'A lady.'
                tokenized_text.append(self._tokenize_text(desc))

                # young
                yi = ATTR.index('Young')
                indd = ind % len(self.attr_dict['cat1'][yi])
                key = self.attr_dict['cat1'][yi][indd]
                frames, _ = self._get_video_by_key(key)
                image_tensor.append(frames)
                desc = 'A person is youthful.'
                tokenized_text.append(self._tokenize_text(desc))

                # bald
                yi = ATTR.index('Bald')
                indd = ind % len(self.attr_dict['cat1'][yi])
                key = self.attr_dict['cat1'][yi][indd]
                frames, _ = self._get_video_by_key(key)
                image_tensor.append(frames)
                desc = 'A person has no hair.'
                tokenized_text.append(self._tokenize_text(desc))

                # eyeglasses
                yi = ATTR.index('Eyeglasses')
                indd = ind % len(self.attr_dict['cat1'][yi])
                key = self.attr_dict['cat1'][yi][indd]
                frames, _ = self._get_video_by_key(key)
                image_tensor.append(frames)
                desc = 'A person wears spectacles.'
                tokenized_text.append(self._tokenize_text(desc))

                # chubby
                yi = ATTR.index('Chubby')
                indd = ind % len(self.attr_dict['cat1'][yi])
                key = self.attr_dict['cat1'][yi][indd]
                frames, _ = self._get_video_by_key(key)
                image_tensor.append(frames)
                desc = 'A person is plump.'
                tokenized_text.append(self._tokenize_text(desc))

                image_tensor = torch.stack(image_tensor)
                tokenized_text = torch.stack(
                    tokenized_text
                ) if self.tokenizer is not None else tokenized_text
                return image_tensor, tokenized_text
            elif self.attr_mode == 'mask':
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
                description = f"A person in image one is talking"
            elif self.attr_mode == 'draw':
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
                description = f"A person in image one is talking"
            elif self.attr_mode == 'mask+text':
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
            elif self.attr_mode == 'mask+text_dropout':
                frame_folder = os.path.join(self.root, 'mask', key)
                if self.deterministic:
                    frame = os.listdir(frame_folder)[0]
                else:
                    frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
                if random.random() < 0.1:
                    description = "null"
            elif self.attr_mode == 'draw+text':
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
            elif self.attr_mode == 'draw+text_dropout':
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual1.unsqueeze(0)
                if random.random() < 0.1:
                    description = "null"
            elif self.attr_mode == 'image_same+draw':
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                if random.random() < 0.5:
                    visuals = torch.stack([visual, visual1], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with appearance in image one and draw in image two is talking"
                    else:
                        description = f"A person with draw in image two and appearance in image one is talking"
                else:
                    visuals = torch.stack([visual1, visual], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with draw in image one and appearance in image two is talking"
                    else:
                        description = f"A person with appearance in image two and draw in image one is talking"
            elif self.attr_mode == 'image_same+mask':
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                if random.random() < 0.5:
                    visuals = torch.stack([visual, visual1], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with appearance in image one and mask in image two is talking"
                    else:
                        description = f"A person with mask in image two and appearance in image one is talking"
                else:
                    visuals = torch.stack([visual1, visual], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with mask in image one and appearance in image two is talking"
                    else:
                        description = f"A person with appearance in image two and mask in image one is talking"
            elif self.attr_mode == 'image+draw':
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                pid = '#'.join(key.split('#')[:2])
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'video', key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                if random.random() < 0.5:
                    visuals = torch.stack([visual2, visual1], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with appearance in image one and draw in image two is talking"
                    else:
                        description = f"A person with draw in image two and appearance in image one is talking"
                else:
                    visuals = torch.stack([visual1, visual2], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with draw in image one and appearance in image two is talking"
                    else:
                        description = f"A person with appearance in image two and draw in image one is talking"
            elif self.attr_mode == 'image+draw2':
                # NOTE: used for testing, makes visualizations easier
                frame_folder = os.path.join(self.root, 'draw', draw_style, key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                pid = '#'.join(key.split('#')[:2])
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'video', key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = torch.stack([visual2, visual1], dim=0)
                if random.random() < 0.5:
                    description = f"A person with appearance in image one and draw in image two is talking"
                else:
                    description = f"A person with draw in image two and appearance in image one is talking"
            elif self.attr_mode == 'image+mask':
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                pid = '#'.join(key.split('#')[:2])
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'video', key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                frame_folder = os.path.join(self.root, 'mask', key_)
                frame = choice(os.listdir(frame_folder))
                visual3 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                if random.random() < 0.5:
                    visuals = torch.stack([visual2, visual1], dim=0)
                    visuals_pos = torch.stack([visual2, visual3], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with appearance in image one and mask in image two is talking"
                    else:
                        description = f"A person with mask in image two and appearance in image one is talking"
                else:
                    visuals = torch.stack([visual1, visual2], dim=0)
                    visuals_pos = torch.stack([visual3, visual2], dim=0)

                    if random.random() < 0.5:
                        description = f"A person with mask in image one and appearance in image two is talking"
                    else:
                        description = f"A person with appearance in image two and mask in image one is talking"
            elif self.attr_mode == 'image+mask2':
                # NOTE: used for testing, makes visualizations easier
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                # pid = key.split('#')[0]
                pid = '#'.join(key.split('#')[:2])
                # key_ = choice(list(set(self.attr_dict['pid'][pid])-set([key])))
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'video', key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                frame_folder = os.path.join(self.root, 'mask', key_)
                frame = choice(os.listdir(frame_folder))
                visual3 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = torch.stack([visual2, visual1], dim=0)
                visuals_pos = torch.stack([visual2, visual3], dim=0)
                if random.random() < 0.5:
                    description = f"A person with appearance in image one and mask in image two is talking"
                else:
                    description = f"A person with mask in image two and appearance in image one is talking"
            elif self.attr_mode == 'draw+mask':
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                pid = '#'.join(key.split('#')[:2])
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'draw', draw_style,
                                            key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                if random.random() < 0.5:
                    visuals = torch.stack([visual2, visual1], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with draw in image one and mask in image two is talking"
                    else:
                        description = f"A person with mask in image two and draw in image one is talking"
                else:
                    visuals = torch.stack([visual1, visual2], dim=0)
                    if random.random() < 0.5:
                        description = f"A person with mask in image one and draw in image two is talking"
                    else:
                        description = f"A person with draw in image two and mask in image one is talking"
            elif self.attr_mode == 'draw+mask2':
                # NOTE: used for testing, makes visualizations easier
                frame_folder = os.path.join(self.root, 'mask', key)
                frame = choice(os.listdir(frame_folder))
                visual1 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                # pid = key.split('#')[0]
                pid = '#'.join(key.split('#')[:2])
                # key_ = choice(list(set(self.attr_dict['pid'][pid])-set([key])))
                key_ = choice(self.attr_dict['pid'][pid])
                frame_folder = os.path.join(self.root, 'draw', draw_style,
                                            key_)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = torch.stack([visual2, visual1], dim=0)
                if random.random() < 0.5:
                    description = f"A person with draw in image one and mask in image two is talking"
                else:
                    description = f"A person with mask in image two and draw in image one is talking"
            elif self.attr_mode == 'image+text_dropout':
                if random.random() < 0.5:  # sample from different sequence
                    pid = '#'.join(key.split('#')[:2])
                    key_ = choice(self.attr_dict['pid'][pid])
                    frame_folder = os.path.join(self.root, 'video', key_)
                else:  # sample from the same sequence
                    frame_folder = os.path.join(self.root, 'video', key)
                if self.deterministic:
                    frame = os.listdir(frame_folder)[0]
                else:
                    frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                visuals = visual2.unsqueeze(0)
                if random.random() < 0.1:
                    description = "null"
            elif self.attr_mode == 'image+video33':
                frame_folder = os.path.join(self.root, 'video', key)
                frame = choice(os.listdir(frame_folder))
                visual2 = self.video_transform(
                    to_tensor(Image.open(os.path.join(frame_folder, frame))))
                # sample motion
                # TODO: hardcoded
                visual_num = 3
                visual_step = 3
                visual3 = image_tensor[:visual_num * visual_step:visual_step,
                                       ...]
                visuals = torch.cat([visual2.unsqueeze(0), visual3], dim=0)
                description = f"A person with appearance in image one and motion in the following frames is talking."
            else:
                visuals = visual.unsqueeze(0)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_neg:
            key_ = self._sample_negative_label(key)
            text_file = Path(os.path.join(self.root, self.texts[key_]))
            descriptions = text_file.read_text().split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            description_ = choice(descriptions)
            tokenized_text_ = self.tokenizer.tokenize(
                description_,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description_
            visual_ = 0
            return tokenized_text, image_tensor, visuals, visual_, tokenized_text_

        return tokenized_text, image_tensor, visuals


class iPERDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        cache=None,
        return_vc=False,
        video_only=False,
        keys=None,
        return_neg=False,
        drop_sentence=False,
        tokenizer2=None,
        rep_num=1,
        slow=False,
        slow_mode=None,
        skip_min_len_check=False,
        return_label=False,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.rep_num = rep_num
        self.slow = slow
        self.slow_mode = slow_mode
        self.image_size = image_size
        self.return_label = return_label

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        frame_step_max = int(self.frame_step *
                             1.5) if slow else self.frame_step
        if skip_min_len_check:
            self.min_len = max(
                min_len, (self.frame_num - 1) * int(self.frame_step * 1.5) + 1)
        else:
            self.min_len = max(min_len,
                               (self.frame_num - 1) * frame_step_max + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_neg = return_neg
        self.video_only = video_only
        self.drop_sentence = drop_sentence

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(
                    tqdm(os.listdir(video_root), desc="Counting videos")):
                key = video  # no stem
                text = key + '.txt'
                if os.path.isdir(os.path.join(video_root,
                                              key)) and text in text_files:
                    frames = natsorted(
                        os.listdir(os.path.join(video_root, key)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(video_root, key,
                                                  frame_name)):
                        frame_list.append(
                            os.path.join('video', key, frame_name))
                if len(frame_list) > 0:
                    # add entry
                    keys_list.append(key)
                    text_dict[key] = os.path.join('txt', text)
                    video_dict[key] = frame_list
                    length_dict[key] = len(frame_list)
                # clear
                frame_list = frames = None
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            assert len(self.keys) > 0
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        if return_neg:
            attr_cache = path.parent / (path.name + '_attr_dict.pkl')
            if attr_cache.exists():
                with open(attr_cache, 'rb') as f:
                    self.attr_dict = pickle.load(f)
            else:
                attr_dict = {'text': {}}
                for k in tqdm(self.keys):
                    descriptions = Path(os.path.join(
                        self.root, self.texts[k])).read_text().split('\n')
                    description = descriptions[0]
                    text = description.lower().replace(',', '')
                    if text in attr_dict['text']:
                        attr_dict['text'][text].append(k)
                    else:
                        attr_dict['text'][text] = [k]
                with open(attr_cache, 'wb') as f:
                    pickle.dump(attr_dict, f)
                self.attr_dict = attr_dict

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = sorted(keys_keep)

        if return_neg:
            attr_dict = {}
            for attr_type in self.attr_dict:
                attr_dict[attr_type] = {}
                for attr in self.attr_dict[attr_type].keys():
                    attr_dict[attr_type][attr] = list(
                        set(self.attr_dict[attr_type][attr]) & set(keys_keep))
            self.attr_dict = attr_dict

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        # image transform
        self.deterministic = deterministic
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_label(self, key):
        label_file = Path(
            os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        label = label_file.read_text().rstrip()
        return int(label)

    def _get_video(self, index, frame_step=None):
        if frame_step is None:
            frame_step = self.frame_step
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = 0 if self.deterministic else random.randint(
            0, video_len - (self.frame_num - 1) * frame_step - 1)  # inclusive
        frames = []
        if self.rep_num == 1:
            frame_idx = range(start_idx,
                              start_idx + self.frame_num * frame_step,
                              frame_step)
        else:
            m_step = int(
                (video_len - (self.frame_num - 1) * frame_step) / self.rep_num)
            frame_idx = []
            for m in range(self.rep_num):
                start_idx = m_step * m
                frame_idx += list(
                    range(start_idx, start_idx + self.frame_num * frame_step,
                          frame_step))
        for i in frame_idx:
            img = Image.open(os.path.join(self.root, self.videos[key][i]))
            img = T.Resize((self.image_size, self.image_size))(img)
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        if True:
            idx = 0 if self.deterministic else random.randint(0, video_len - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frames, key, visual
        return frames, key

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        frame = Image.open(os.path.join(self.root,
                                        self.videos[key][frame_idx]))
        frame = self.image_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frame, key, visual
        return frame, key

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        key = self.keys[video_id]
        frame = Image.open(os.path.join(self.root, self.videos[key][frame_id]))
        frame = self.image_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, key

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        key = self.keys[video_id]
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(
                os.path.join(self.root, self.videos[key][frame_id + i]))
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        return frames, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        frame_step, slow_desc = None, ''
        if self.slow:
            if self.deterministic:
                if self.slow_mode is None:
                    num = 1
                elif self.slow_mode == 'slow':
                    num = 0
                elif self.slow_mode == 'normal':
                    num = 1
                elif self.slow_mode == 'fast':
                    num = 2
                else:
                    raise NotImplementedError
            else:
                num = random.randint(0, 2)
            if num == 0:  # slow
                frame_step = self.frame_step // 2
                slow_desc = 'slow speed.'
            elif num == 1:  # normal
                frame_step = self.frame_step
                slow_desc = 'normal speed.'
            elif num == 2:  # fast
                frame_step = self.frame_step + self.frame_step // 2
                slow_desc = 'fast speed.'

        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind, frame_step)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.video_only:
            description = 'dummy text'
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description
            if self.return_label:
                label = self._get_label(key)
                return tokenized_text, image_tensor, label
            return tokenized_text, image_tensor, visual

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
            if self.drop_sentence:
                # e.g.: "person 001 dressed in 10 is performing random pose."
                if self.deterministic:
                    description = description[:-1] + ','
                else:
                    xxx = description.split(' ')[1]
                    yyy = description.split(' ')[4]
                    zzz = description.split(' ')[7]
                    xxx = 'a person' if random.random(
                    ) < 0.5 else f"person {xxx}"
                    yyy = '' if random.random() < 0.1 else f"dressed in {yyy}"
                    pose = f"'A' pose" if zzz == "'A'" else "random pose"
                    zzz = 'is performing some pose' if random.random(
                    ) < 0.5 else f"is performing {pose}"
                    description = f"{xxx} {yyy} {zzz},"
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        if self.slow:
            description = description + ' ' + slow_desc

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_neg:
            text = descriptions[0].lower().replace(',', '')
            text_ = choice(
                list(set(self.attr_dict['text'].keys()) - set([text])))
            key_ = choice(self.attr_dict['text'][text_])
            text_file = Path(os.path.join(self.root, self.texts[key_]))
            descriptions = text_file.read_text().split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            description_ = choice(descriptions)
            tokenized_text_ = self.tokenizer.tokenize(
                description_,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description_
            visual_ = 0
            return tokenized_text, image_tensor, visual, visual_, tokenized_text_

        return tokenized_text, image_tensor, visual


class ShapeDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        image_only=False,
        cache=None,
        return_vc=False,
        keys=None,
        no_cache=False,
    ):
        super().__init__()
        self.mode = mode
        self.mode == 'video'
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.deterministic = deterministic

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.min_len = max(min_len, (self.frame_num - 1) * self.frame_step + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.image_only = image_only

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        cache = path.parent / (path.name +
                               '_local.db') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            video_list = []
            length_list = []
            text_list = []
            for i, video in enumerate(
                    tqdm(os.listdir(video_root), desc="Counting videos")):
                text = video + '.txt'
                if os.path.isdir(os.path.join(video_root,
                                              video)) and text in text_files:
                    frames = natsorted(
                        os.listdir(os.path.join(video_root, video)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(
                            os.path.join(video_root, video, frame_name)):
                        # do not include dataroot here so that cache can be shared
                        frame_list.append(
                            os.path.join('video', video, frame_name))
                if len(frame_list) >= min_len:
                    video_list.append(frame_list)
                    length_list.append(len(frame_list))
                    text_list.append(os.path.join('txt', text))
                frame_list = frames = None
            self.texts, self.videos, self.lengths = text_list, video_list, length_list
            if not no_cache and cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)
        # filter out videos that are too short
        inds = [i for i, l in enumerate(self.lengths) if l >= self.min_len]
        if keys is not None:
            keys_all = [Path(k).stem for k in self.texts]
            inds = set(inds) & set(
                [keys_all.index(k) for k in (set(keys) & set(keys_all))])
        self.texts = [self.texts[i] for i in inds]
        self.videos = [self.videos[i] for i in inds]
        self.lengths = [self.lengths[i] for i in inds]
        self.keys = [Path(k).stem for k in self.texts]

        self.cumsum = np.cumsum([0] + self.lengths)
        self.lengthsn = [i - nframe_num + 1 for i in self.lengths]
        self.cumsumn = np.cumsum([0] + self.lengthsn)

        if self.mode == 'video':
            self._dataset_length = len(self.videos)
        elif self.mode == '1frame':
            self._dataset_length = len(self.videos)
        elif self.mode == 'image':
            self._dataset_length = np.sum(self.lengths)
        elif self.mode == 'nframe':
            self._dataset_length = np.sum(self.lengthsn)
        else:
            raise NotImplementedError

        # image transform
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_video(self, index):
        video_len = self.lengths[index]
        if self.deterministic:
            start_idx = 0
        else:
            start_idx = random.randint(0, video_len -
                                       (self.frame_num - 1) * self.frame_step -
                                       1)  # inclusive
        frames = []
        for i in range(start_idx, start_idx + self.frame_num * self.frame_step,
                       self.frame_step):
            img = Image.open(os.path.join(self.root, self.videos[index][i]))
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        if True:
            if self.deterministic:
                idx = random.randint(0, video_len - 1)
            else:
                idx = video_len // 2
            visual = Image.open(
                os.path.join(self.root, self.videos[index][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frames, index, visual
        return frames, index

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        video_len = self.lengths[index]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        frame = Image.open(
            os.path.join(self.root, self.videos[index][frame_idx]))
        frame = self.image_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.open(
                os.path.join(self.root, self.videos[index][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frame, index, visual
        return frame, index

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        frame = Image.open(
            os.path.join(self.root, self.videos[video_id][frame_id]))
        frame = self.image_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, video_id

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(
                os.path.join(self.root, self.videos[video_id][frame_id + i]))
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        return frames, video_id

    def _get_label(self, key):
        label_path = os.path.join(self.root,
                                  self.texts[key]).replace('txt/', 'label/')
        label = [int(s) for s in Path(label_path).read_text().split(',')]
        return np.array(label)

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        visual = 0

        image_tensor, key, visual = self._get_video(ind)

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        # Success
        label = self._get_label(key)
        visuals = visual.unsqueeze(0)
        return tokenized_text, image_tensor, visuals, label


class ShapeAttrDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        cache=None,
        return_vc=False,
        keys=None,
        attr_mode=1,
        return_neg=False,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.attr_mode = attr_mode
        self.deterministic = deterministic

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.min_len = max(min_len, (self.frame_num - 1) * self.frame_step + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_neg = return_neg

        assert (path.parent / (path.name + '_attr_dict.pkl')).exists()
        with open(path.parent / (path.name + '_attr_dict.pkl'), 'rb') as f:
            self.attr_dict = pickle.load(f)

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        self.has_label = (Path(dataroot) / 'label').exists()
        self.has_visual = (Path(dataroot) / 'visual').exists()
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(
                    tqdm(os.listdir(video_root), desc="Counting videos")):
                key = Path(video).stem
                text = key + '.txt'
                if os.path.isdir(os.path.join(video_root,
                                              key)) and text in text_files:
                    frames = natsorted(
                        os.listdir(os.path.join(video_root, key)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(video_root, key,
                                                  frame_name)):
                        frame_list.append(
                            os.path.join('video', key, frame_name))
                if len(frame_list) >= 1:
                    # add entry
                    keys_list.append(key)
                    text_dict[key] = os.path.join('txt', text)
                    video_dict[key] = frame_list
                    length_dict[key] = len(frame_list)
                # clear
                frame_list = frames = None
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = keys_keep

        attr_dict = {}
        for attr_type in self.attr_dict:
            attr_dict[attr_type] = {}
            for attr in self.attr_dict[attr_type].keys():
                attr_dict[attr_type][attr] = list(
                    set(self.attr_dict[attr_type][attr]) & set(keys_keep))
        self.attr_dict = attr_dict

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        # image transform
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_video(self, index):
        key = self.keys[index]
        video_len = self.lengths[key]
        if self.deterministic:
            start_idx = 0
        else:
            start_idx = random.randint(0, video_len -
                                       (self.frame_num - 1) * self.frame_step -
                                       1)  # inclusive
        frames = []
        for i in range(start_idx, start_idx + self.frame_num * self.frame_step,
                       self.frame_step):
            img = Image.open(os.path.join(self.root, self.videos[key][i]))
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        if True:
            if self.deterministic:
                idx = 0
            else:
                idx = random.randint(0, video_len - 1)
            visual_path = os.path.join(self.root, self.videos[key][idx])
            visual = Image.open(
                visual_path.replace('video/', 'visual/') if self.
                has_visual else visual_path)
            visual = self.image_transform(to_tensor(visual))
            return frames, key, visual
        return frames, key

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        frame = Image.open(os.path.join(self.root,
                                        self.videos[key][frame_idx]))
        frame = self.image_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frame, key, visual
        return frame, key

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        key = self.keys[video_id]
        frame = Image.open(os.path.join(self.root, self.videos[key][frame_id]))
        frame = self.image_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, key

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        key = self.keys[video_id]
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(
                os.path.join(self.root, self.videos[key][frame_id + i]))
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        return frames, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        visual = 0
        visual1, visual2, visual3 = 0, 0, 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
        else:
            raise RuntimeError

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
            size, color, shape = description.split(' is moving')[0][2:].split()
            motion = description.split(' is moving ')[1]
            if self.attr_mode == 'text':
                visuals = visual.unsqueeze(0)
            elif self.attr_mode == 'object':
                obj = f'{size} {color} {shape}'
                key_attr = choice(self.attr_dict['object'][obj])
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual1 = Image.open(
                    os.path.join(self.root, self.videos[key_attr][idx]))
                visual1 = self.image_transform(to_tensor(visual1))
                visuals = visual1
                description = f'An object in image one is moving {motion}'
            elif self.attr_mode == 'object_same':
                visuals = visual
                description = f'An object in image one is moving {motion}'
            elif self.attr_mode == 'object+same_background':
                obj = f'{size} {color} {shape}'
                key_attr = choice(self.attr_dict['object'][obj])
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual1 = Image.open(
                    os.path.join(self.root, self.videos[key_attr][idx]))
                visual1 = self.image_transform(to_tensor(visual1))
                visual2 = visual
                visuals = torch.stack([visual1, visual2], dim=0)
                description = f'An object in image one with background in image two is moving {motion}'
            elif self.attr_mode == 'object+same_background+rand':
                obj = f'{size} {color} {shape}'
                key_attr = choice(self.attr_dict['object'][obj])
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual1 = Image.open(
                    os.path.join(self.root, self.videos[key_attr][idx]))
                visual1 = self.image_transform(to_tensor(visual1))
                visual2 = visual
                if random.random() < 0.5:
                    visuals = torch.stack([visual1, visual2], dim=0)
                    description = f'An object in image one with background in image two is moving {motion}'
                else:
                    visuals = torch.stack([visual2, visual1], dim=0)
                    description = f'An object in image two with background in image one is moving {motion}'
            elif self.attr_mode == 'same_object+same_background':
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual2 = Image.open(
                    os.path.join(self.root, self.videos[key][idx]))
                visual2 = self.image_transform(to_tensor(visual2))
                visuals = torch.stack([visual, visual2], dim=0)
                description = f'An object in image one with background in image two is moving {motion}'
            elif self.attr_mode == 'color+shape+background+rand':
                key_color = choice(self.attr_dict['color'][color])
                key_shape = choice(self.attr_dict['shape'][shape])
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual1 = Image.open(
                    os.path.join(self.root, self.videos[key_color][idx]))
                visual1 = self.image_transform(to_tensor(visual1))
                visual2 = Image.open(
                    os.path.join(self.root, self.videos[key_shape][idx]))
                visual2 = self.image_transform(to_tensor(visual2))
                visual3 = visual
                if random.random() < 0.5:
                    visual_order = 123
                    visuals = torch.stack([visual1, visual2, visual3], dim=0)
                    if random.random() < 0.5:
                        description = f'An object with color in image one, shape in image two, background in image three is moving {motion}'
                        description_ = f'An object with color in image two, shape in image one, background in image three is moving {motion}'
                    else:
                        description = f'An object with shape in image two, color in image one, background in image three is moving {motion}'
                        description_ = f'An object with shape in image one, color in image two, background in image three is moving {motion}'
                else:
                    visual_order = 213
                    visuals = torch.stack([visual2, visual1, visual3], dim=0)
                    if random.random() < 0.5:
                        description = f'An object with shape in image one, color in image two, background in image three is moving {motion}'
                        description_ = f'An object with shape in image two, color in image one, background in image three is moving {motion}'
                    else:
                        description = f'An object with color in image two, shape in image one, background in image three is moving {motion}'
                        description_ = f'An object with color in image one, shape in image two, background in image three is moving {motion}'
                if self.return_neg:
                    idx = random.randint(0, image_tensor.shape[0] - 1)
                    color_ = choice(
                        list(
                            set(self.attr_dict['color'].keys()) -
                            set([color])))
                    # key_color_ = choice(self.attr_dict['color'][color_])
                    key_color_ = choice(
                        list(
                            set(self.attr_dict['color'][color_]) -
                            set(self.attr_dict['shape'][shape])))
                    shape_ = choice(
                        list(
                            set(self.attr_dict['shape'].keys()) -
                            set([shape])))
                    # key_shape_ = choice(self.attr_dict['shape'][shape_])
                    key_shape_ = choice(
                        list(
                            set(self.attr_dict['shape'][shape_]) -
                            set(self.attr_dict['color'][color])))
                    key_ = choice(list(set(self.keys) - set([key])))
                    visual1_ = Image.open(
                        os.path.join(self.root, self.videos[key_color_][idx]))
                    visual1_ = self.image_transform(to_tensor(visual1_))
                    visual2_ = Image.open(
                        os.path.join(self.root, self.videos[key_shape_][idx]))
                    visual2_ = self.image_transform(to_tensor(visual2_))
                    visual3_ = Image.open(
                        os.path.join(self.root,
                                     self.videos[key_][idx]).replace(
                                         'video/', 'visual/'))
                    visual3_ = self.image_transform(to_tensor(visual3_))
                    if visual_order == 123:
                        visuals_ = torch.stack([visual1_, visual2_, visual3_],
                                               dim=0)
                    elif visual_order == 213:
                        visuals_ = torch.stack([visual2_, visual1_, visual3_],
                                               dim=0)
                    tokenized_text_ = self.tokenizer.tokenize(
                        description_,
                        self.text_len,
                        truncate_text=self.truncate_captions,
                    ).squeeze(
                        0) if self.tokenizer is not None else description_
            elif self.attr_mode == 'color+shape+background':
                key_color = choice(self.attr_dict['color'][color])
                key_shape = choice(self.attr_dict['shape'][shape])
                idx = random.randint(0, image_tensor.shape[0] - 1)
                visual1 = Image.open(
                    os.path.join(self.root, self.videos[key_color][idx]))
                visual1 = self.image_transform(to_tensor(visual1))
                visual2 = Image.open(
                    os.path.join(self.root, self.videos[key_shape][idx]))
                visual2 = self.image_transform(to_tensor(visual2))
                visual3 = visual
                visuals = torch.stack([visual1, visual2, visual3], dim=0)
            else:
                raise NotImplementedError

        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_neg:
            return tokenized_text, image_tensor, visuals, visuals_, tokenized_text_
        return tokenized_text, image_tensor, visuals
