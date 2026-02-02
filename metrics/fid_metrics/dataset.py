import bisect
import glob
from typing import List
import os

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    'ImageDataset',
    'ImageSequenceDataset',
    'VideoDataset',
    'is_image_dir_path',
    'is_video_path',
]


class SequeceTransform:
    def __init__(self, transforms, stack_dim=1):
        self.transforms = transforms
        self.stack_dim = stack_dim

    def __call__(self, imgs: List) -> torch.Tensor:
        imgs = [self.transforms(img) for img in imgs]
        return torch.stack(imgs, dim=self.stack_dim) if len(imgs) > 1 else imgs[0]


class ImageDataset(Dataset):
    def __init__(self, image_dirs, resize_shape=(256, 512), ext='png'):
        self.image_dirs = glob.glob(image_dirs)
        self.transforms = T.Compose([T.ToTensor(), T.Resize(resize_shape)])

        self.files = []
        for image_dir in self.image_dirs:
            imgs = glob.glob(image_dir + (f'/*.{ext}' if ext else '/*'))
            self.files.extend(imgs)
        print(f'Loaded {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        image_dirs,
        sequence_length=16,
        resize_shape=(224, 224),
        ext='png',
        no_overlap=True,
    ):
        self.image_dirs = glob.glob(image_dirs)
        self.sequence_length = sequence_length
        self.no_overlap = no_overlap
        self.transforms = SequeceTransform(T.Compose([
            T.ToTensor(), 
            T.Resize(resize_shape),
        ]))

        self.frame_paths = []
        for image_dir in self.image_dirs:
            frame_paths = sorted(glob.glob(image_dir + (f'/*.{ext}' if ext else '/*')), key=lambda x: int(os.path.basename(x).split('.')[0].split('frame_')[-1]))
            self.frame_paths.extend(frame_paths)
        print(f'Loaded {len(self.frame_paths)} images')

    def __len__(self):
        return (
            len(self.frame_paths) // self.sequence_length
            if self.no_overlap
            else len(self.frame_paths)
        )

    def __getitem__(self, idx):
        if self.no_overlap:
            frame_paths = self.frame_paths[
                idx * self.sequence_length : (idx + 1) * self.sequence_length
            ]
        else:
            frame_paths = self.frame_paths[idx : idx + self.sequence_length]

        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        frames = self.transforms(frames)
        return frames


class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=16, resize_shape=(224, 224), no_overlap=True):
        self.video_paths = glob.glob(video_path)
        self.sequence_length = sequence_length
        self.no_overlap = no_overlap
        self.transforms = SequeceTransform(T.Compose([T.ToTensor(), T.Resize(resize_shape)]))

        total_frames = 0
        self.num_accum_sequences = []
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            total_frames += num_frames
            self.num_accum_sequences.append(
                (total_frames // sequence_length)
                if no_overlap
                else (total_frames - sequence_length + 1)
            )
            cap.release()
        print(f'Loaded {len(self.video_paths)} video files, {total_frames} frames total')

    def __len__(self):
        return self.num_accum_sequences[-1]

    def __getitem__(self, idx):
        video_idx = bisect.bisect_left(self.num_accum_sequences, idx)
        video_path = self.video_paths[video_idx]
        frame_idx = idx - self.num_accum_sequences[video_idx]
        if self.no_overlap:
            frame_idx *= self.sequence_length

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            assert ret
            frames.append(frame)
        cap.release()

        frames = self.transforms(frames)
        return frames


def is_image_dir_path(path, ext='png'):
    return len(glob.glob(path + (f'/*.{ext}' if ext else '/*'))) > 0


def is_video_path(path, ext='mp4'):
    return glob.glob(path)[0].endswith(f'.{ext}')