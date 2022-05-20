from pathlib import Path
import random
from random import randint, choice
import os
import pickle
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
import decord

decord.bridge.set_bridge("torch")

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

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


class TextImageDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        cache=None,
        image_only=False,
        deterministic=False,
    ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.image_only = image_only
        path = Path(folder)

        cache = path.parent / (path.name +
                               '_local.db') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                self.keys, self.text_files, self.image_files = pickle.load(f)
        else:
            text_files = [*path.glob('**/*.txt')]
            image_files = [
                *path.glob('**/*.png'), *path.glob('**/*.jpg'),
                *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
            ]

            text_files = {
                text_file.stem: text_file
                for text_file in text_files
            }
            image_files = {
                image_file.stem: image_file
                for image_file in image_files
            }

            keys = (image_files.keys() & text_files.keys())

            self.keys = list(keys)
            self.text_files = {
                k: v
                for k, v in text_files.items() if k in keys
            }
            self.image_files = {
                k: v
                for k, v in image_files.items() if k in keys
            }
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.keys, self.text_files, self.image_files),
                                f)

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        if deterministic:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor()
            ])
        else:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.keys)

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
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description, self.text_len, truncate_text=self.truncate_captions
        ).squeeze(0) if self.tokenizer is not None else description
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError,
                OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        if self.image_only:
            return image_tensor, 0

        # Success
        return tokenized_text, image_tensor


class TextVideoDataset(Dataset):
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
        self.image_size = image_size
        self.return_label = return_label

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        frame_step_max = self.frame_step
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
        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
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
                description_ = description.split('. ')
                if self.deterministic:
                    description = description_[0]
                    if 'and' in description:
                        description = description.split(', ')[0] + '.'
                else:
                    # num_drop = random.randint(0, min(len(description_)-1, 3))
                    num_drop = random.randint(0, len(description_) - 1)
                    for _ in range(num_drop):
                        description_.remove(random.choice(description_))
                    description = '. '.join(description_)
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


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen,
                            num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames,
                               vlen,
                               sample=sample,
                               fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


class TextMP4Dataset(Dataset):
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
        return_text=False,
        return_label=False,
        keys=None,
        video_only=False,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

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
        self.return_text = return_text
        self.return_label = return_label
        self.image_only = image_only
        self.video_only = video_only

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        self.has_label = (Path(dataroot) / 'label').exists()
        self.has_visual = (Path(dataroot) / 'visual').exists()

        # Build or load cache
        video_files = os.listdir(video_root)
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
            for i, video in enumerate(tqdm(video_files,
                                           desc="Counting videos")):
                videoid = Path(video).stem
                text = videoid + '.txt'
                if is_video_file(video) and text in text_files:
                    # get video info
                    video_path = os.path.join(self.root, 'video', video)
                    try:
                        video_reader = decord.VideoReader(video_path,
                                                          num_threads=1)
                        vlen = len(video_reader)
                        # add entry
                        keys_list.append(videoid)
                        text_dict[videoid] = os.path.join('txt', text)
                        video_dict[videoid] = os.path.join('video', video)
                        length_dict[videoid] = vlen
                    except:
                        continue
                else:
                    continue
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

    def _get_label(self, key):
        label_file = Path(
            os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        label = label_file.read_text().rstrip()
        return int(label)

    def _get_video(self, index):
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = random.randint(0, video_len -
                                   (self.frame_num - 1) * self.frame_step -
                                   1)  # inclusive
        video_path = os.path.join(self.root, self.videos[key])
        video_reader = decord.VideoReader(video_path, num_threads=1)
        frame_idxs = range(start_idx,
                           start_idx + self.frame_num * self.frame_step,
                           self.frame_step)
        frames = video_reader.get_batch(frame_idxs)
        frames = frames.float() / 255  # to [0, 1]
        frames = frames.permute(0, 3, 1, 2)
        frames = self.image_transform(frames)
        if True:
            idx = random.randint(0, video_len - 1)
            visual = video_reader.get_batch([idx])
            visual = visual.permute(0, 3, 1, 2).squeeze().float() / 255
            visual = self.image_transform(visual)
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
        video_path = os.path.join(self.root, self.videos[key])
        video_reader = decord.VideoReader(video_path, num_threads=1)
        frame = video_reader.get_batch([frame_idx])
        frame = frame.permute(0, 3, 1, 2).squeeze().float() / 255
        frame = self.image_transform(frame)
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = video_reader.get_batch([idx])
            visual = visual.permute(0, 3, 1, 2).squeeze().float() / 255
            visual = self.image_transform(visual)
            return frame, key, visual
        return frame, key

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
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        # if self.image_only:
        #     return image_tensor, 0
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
        try:
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

        if self.return_label:
            label = self._get_label(key)
            return tokenized_text, image_tensor, label

        # Success
        # if self.return_vc:
        #     return tokenized_text, image_tensor, visual
        # if self.return_text:
        #     return tokenized_text, image_tensor, description
        # return tokenized_text, image_tensor

        return tokenized_text, image_tensor, visual


class TextImageStackDataset(Dataset):
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
        return_text=False,
        return_label=False,
        keys=None,
        no_cache=False,
    ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

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
        self.return_text = return_text
        self.return_label = return_label
        self.image_only = image_only

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        self.has_label = (Path(dataroot) / 'label').exists()
        self.has_visual = (Path(dataroot) / 'visual').exists()

        # Build or load cache
        video_files = os.listdir(video_root)
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
            for i, video in enumerate(tqdm(video_files,
                                           desc="Counting videos")):
                videoid = Path(video).stem
                text = videoid + '.txt'
                if is_image_file(video) and text in text_files:
                    # get video info
                    video_path = os.path.join(self.root, 'video', video)
                    try:
                        imgs = Image.open(video_path).convert('RGB')
                        imgs = np.array(imgs)  # [H, W, C]
                        # horizontal = imgs.shape[1] > imgs.shape[0]
                        shorter, longer = min(imgs.shape[0],
                                              imgs.shape[1]), max(
                                                  imgs.shape[0], imgs.shape[1])
                        vlen = longer // shorter
                        # frames = np.split(imgs, vlen, axis=1 if horizontal else 0)

                        # add entry
                        keys_list.append(videoid)
                        text_dict[videoid] = os.path.join('txt', text)
                        video_dict[videoid] = os.path.join('video', video)
                        length_dict[videoid] = vlen
                    except:
                        continue
                else:
                    continue
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            if not no_cache and cache is not None:
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
        start_idx = random.randint(0, video_len -
                                   (self.frame_num - 1) * self.frame_step -
                                   1)  # inclusive
        frame_idxs = range(start_idx,
                           start_idx + self.frame_num * self.frame_step,
                           self.frame_step)
        video_path = os.path.join(self.root, self.videos[key])
        frames = read_frames_imagestack(video_path, frame_idxs)
        image_tensor = self.image_transform(frames)
        if not self.return_vc:
            return image_tensor, key, None
        if self.has_visual:
            idx = random.randint(0, video_len - 1)
            visual_path = os.path.join(self.root, 'visual',
                                       Path(self.videos[key]).name)
            visuals = read_frames_imagestack(visual_path, [idx])
            visual_tensor = self.image_transform(visuals[0])
        else:
            idx = random.randint(0, video_len - 1)
            visual = frames[idx]
            visual_tensor = self.image_transform(visual)
        return image_tensor, key, visual_tensor

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        video_path = os.path.join(self.root, self.videos[key])
        frames = read_frames_imagestack(video_path, None)
        frame = frames[frame_idx]
        image_tensor = self.image_transform(frame)
        if not self.return_vc:
            return image_tensor, key, None
        if self.has_visual:
            idx = random.randint(0, video_len - 1)
            visual_path = os.path.join(self.root, 'visual',
                                       Path(self.videos[key]).name)
            visuals = read_frames_imagestack(visual_path, [idx])
            visual_tensor = self.image_transform(visuals[0])
        else:
            idx = random.randint(0, video_len - 1)
            visual = frames[idx]
            visual_tensor = self.image_transform(visual)
        return image_tensor, key, visual_tensor

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
        label_path = os.path.join(self.root,
                                  self.texts[key]).replace('txt/', 'label/')
        label = [int(s) for s in Path(label_path).read_text().split(',')]
        return np.array(label)

    def __getitem__(self, ind):
        visual_tensor = 0
        if self.mode == 'video':
            image_tensor, key, visual_tensor = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual_tensor = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.image_only:
            return image_tensor, 0

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
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
        if self.return_label:
            label = self._get_label(key)
            return tokenized_text, image_tensor, label
        if self.return_vc:
            return tokenized_text, image_tensor, visual_tensor
        if self.return_text:
            return tokenized_text, image_tensor, description
        return tokenized_text, image_tensor
