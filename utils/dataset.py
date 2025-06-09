import os, math, random
import numpy as np

import torch
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as F
import torch.distributed as dist
import math
import noise


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def gen_mpi_mask(D, f, segments=4, alpha=2.0):
    diff = torch.abs(D - f)
    sharpness = f ** alpha if f > 0 else 1e-6
    base = torch.linspace(0, 1, segments + 1, device=D.device)
    thresholds = base ** (1 / sharpness)
    region_mask = torch.bucketize(diff, thresholds[1:-1])
    return region_mask

class BokehCocMPIDatasetWholeVid(Dataset):
    def __init__(
            self,
            csv_path,
            sample_size=(576,1024),
            concept_num=4,
            overlap_frames=4,
            group_frames=8,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        df = pd.read_csv(csv_path)
        self.video_folder = df['aif_folder'].tolist()
        self.depth_folder = df['disp_folder'].tolist()
        self.K = df['k'].tolist()
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.length = len(self.video_folder)
        self.sample_size = sample_size
        self.concept_num = concept_num
        self.overlap_frames = overlap_frames
        self.group_frames = group_frames
    
    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]

    def group_frames_with_overlap(self, files, weight_function=None):
        if not weight_function:
            def weight_function(position, overlap):
                return 1.0 - position / overlap
        
        groups = []
        weights = {}
        group_weights = []
        
        if len(files) <= self.group_frames:
            groups.append(files)
            for file in files:
                weights[file] = 1.0
            return groups, weights
        
        step = self.group_frames - self.overlap_frames
        num_groups = (len(files) - self.overlap_frames) // step
        if (len(files) - self.overlap_frames) % step != 0:
            num_groups += 1
        
        for i in range(num_groups):
            start_idx = i * step
            end_idx = min(start_idx + self.group_frames, len(files))
            group = files[start_idx:end_idx]
            groups.append(group)
            
            for j, file in enumerate(group):
                if i > 0 and j < self.overlap_frames:
                    overlap_weight = weight_function(j, self.overlap_frames)
                    weights[file] = overlap_weight
                else:
                    weights[file] = 1.0
        
        for i, group in enumerate(groups):
            group_w = []
            for j, item in enumerate(group):
                if i > 0 and j < self.overlap_frames:
                    group_w.append(1 - weights.get(item))
                else:
                    group_w.append(weights.get(item))
            group_weights.append(group_w)
        return groups, group_weights

    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[0].split('.')[0])
        def cosine_weight(position, overlap):
            return 0.5 * (1 + math.cos(math.pi * position / overlap))
    
        video_folder = self.video_folder[idx]
        depth_folder = self.depth_folder[idx]

        image_files = sorted(os.listdir(video_folder), key=sort_frames)
        depth_files = sorted(os.listdir(depth_folder), key=sort_frames)

        image_groups, image_weights = self.group_frames_with_overlap(image_files, cosine_weight)
        depth_groups, _ = self.group_frames_with_overlap(depth_files, cosine_weight)

        # Load frame
        group_res = []
       
        for group_i, group_d, weight in zip(image_groups, depth_groups, image_weights):
            numpy_image = np.array([pil_image_to_numpy(Image.open(os.path.join(video_folder, img))) for img in group_i])

            pixel_values = numpy_to_pt(numpy_image)

            # Load frames
            f_list = []
            k_list = []

            for i in range(len(group_i)):
                if self.K[idx] == 'change':
                    k_list.append(float(group_d[i].split('_k_')[1].replace('.png', '')))
                    f_list.append(float(group_d[i].split('zf_')[1].split('_k_')[0]))
                else:
                    f_list.append(float(group_d[i].split('zf_')[1].replace('.png', '')))
                    k_list.append(float(self.K[idx]))


            # Load depth frames
            mpi_mask_list = []
            coc_list = []
            coc_list_ori = []
            for i in range(len(group_i)):
                disparity = np.array(Image.open(os.path.join(depth_folder, group_d[i])).convert("L"))
                disparity = disparity.astype(np.float64)

                if not isinstance(disparity, torch.Tensor):
                    disparity = torch.from_numpy(disparity).float()

                coc_r = torch.abs(disparity - f_list[i] * 255) / 255  # norm
                mpi_mask = gen_mpi_mask(disparity/255, f_list[i], alpha=1.5)

                mpi_mask_list.append(mpi_mask)

                coc_cond = coc_r.repeat(3, 1, 1)
                coc_list.append(coc_cond)

                coc_list_ori.append(Image.fromarray((coc_r * 255).int().cpu().numpy()))
            
            coc_pixel_values = torch.stack(coc_list)
            mpi_mask = torch.stack(mpi_mask_list)  # [4*f, h, w]
            # Load motion values
            motion_values = 127

            pixel_values = self.pixel_transforms(pixel_values)
            coc_pixel_values = self.pixel_transforms(coc_pixel_values)
            mpi_mask = F.resize(mpi_mask, self.sample_size)

            if len(group_i) != self.group_frames:
                pixel_values_pad = torch.zeros_like(group_res[-1]["pixel_values"])
                coc_pixel_values_pad = torch.zeros_like(group_res[-1]["coc"])
                k_pad = torch.zeros_like(group_res[-1]["k"])
                mpi_mask_pad = torch.zeros_like(group_res[-1]["mpi_mask"])
                weight_pad = torch.zeros_like(group_res[-1]["weight"])

                pixel_values_pad[:len(group_i)] = pixel_values
                coc_pixel_values_pad[:len(group_i)] = coc_pixel_values
                k_pad[:len(group_i)] = torch.asarray(k_list)
                mpi_mask_pad[:len(group_i)] = mpi_mask
                weight_pad[:len(group_i)] = torch.asarray(weight)
                group = dict(pixel_values=pixel_values_pad, coc=coc_pixel_values_pad, motion_values=motion_values, k=k_pad, mpi_mask=mpi_mask_pad, weight=weight_pad)
            else:
                group = dict(pixel_values=pixel_values, coc=coc_pixel_values, motion_values=motion_values, k=torch.asarray(k_list), mpi_mask=mpi_mask, weight=torch.asarray(weight))
            
            group_res.append(group)
        return group_res

    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.get_batch(idx) 
        return sample


class BokehCocMPIDataset(Dataset):
    def __init__(
            self,
            csv_path,
            sample_size=(576,1024),
            sample_n_frames=14,
            return_ori_data=False,
            aug_depth=False,
            aug_depth_p=0.0,
            concept_num=4,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        df = pd.read_csv(csv_path)
        self.video_folder = df['aif_folder'].tolist()
        self.bokeh_folder = df['bokeh_folder'].tolist()
        self.depth_folder = df['disp_folder'].tolist()
        self.mask_folder = df['mask_folder'].tolist()
        self.K = df['k'].tolist()
        self.sample_n_frames = sample_n_frames
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # norm 到-1，1
        ])
        self.length = len(self.video_folder)
        self.return_ori_data = return_ori_data
        self.concept_i = df['concept_i'].tolist()
        self.sample_size = sample_size
        self.aug_depth = aug_depth
        self.concept_num = concept_num
        if aug_depth:
            print('aug_depth')
            print(f'aug_depth_p--{aug_depth_p}')
            self.aug_depth_p = aug_depth_p
            self.elastic = transforms.ElasticTransform(alpha=50.0, sigma=5.0)

    
    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
    
    def jagged_edges(self, depth_img):
        rand_kernel = random.choice([3, 5, 7])
        kernel = np.ones((rand_kernel, rand_kernel), np.uint8)
        rand_num = random.random() 
        # random chose dilate or erode
        if rand_num <= self.aug_depth_p / 2:
            depth = cv2.dilate(depth_img.numpy(), kernel)
        elif self.aug_depth_p / 2 < rand_num < self.aug_depth_p  :
            depth = cv2.erode(depth_img.numpy(), kernel)
        else:
            depth = depth_img.numpy()
        return torch.from_numpy(depth)
    
    def gaussian_blur(self, image, kernel_size=5, sigma=2.0):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image.numpy(), (kernel_size, kernel_size), sigma)
        return torch.from_numpy(blurred)


    def generate_multi_perlin_noise(self, scales, amplitudes, mask=None):
        """
        Generate multi-scale Perlin noise
        
        params:
        - scales: Perlin noise scales
        - amplitudes: amplitudes list
        - mask: forground mask
        
        """
        height, width = 576, 1024
        noise_map = np.zeros((height, width))
        
        if mask is not None:
            if mask.shape != (height, width):
                raise ValueError(f"The mask shape should be ({height}, {width}), but the shape is {mask.shape}")
            
            mask_indices = np.where(mask > 0)
            coords = list(zip(mask_indices[0], mask_indices[1]))
        else:
            coords = [(i, j) for i in range(height) for j in range(width)]
        
        offset_x = np.random.randint(0, 10000)
        offset_y = np.random.randint(0, 10000)
        
        for scale, amp in zip(scales, amplitudes):
            # Perlin noise
            for i, j in coords:
                val = noise.pnoise2((i + offset_x) / scale, (j + offset_y) / scale, octaves=3)
                noise_map[i, j] += val * amp
        
        if mask is not None:
            noise_map = noise_map * mask
        
        if noise_map.max() == noise_map.min():
            return torch.zeros((height, width), dtype=torch.float32)
        
        # norm
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        return torch.from_numpy(noise_map).float()

    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[0].split('.')[0])
    
        # Sort and limit the number of image and depth files to sample_n_frames
        video_folder = self.video_folder[idx]
        depth_folder = self.depth_folder[idx]
        bokeh_folder = self.bokeh_folder[idx]
        mask_folder = self.mask_folder[idx]

        image_files = sorted(os.listdir(video_folder), key=sort_frames)
        depth_files = sorted(os.listdir(depth_folder), key=sort_frames)
        bokeh_files = sorted(os.listdir(bokeh_folder), key=sort_frames)
        concept_idx = self.concept_i[idx]


        numbers = list(range(len(os.listdir(video_folder)) - self.sample_n_frames + 1))
        random_numbers = random.sample(numbers, 1)[0]
        image_files = image_files[random_numbers: random_numbers + self.sample_n_frames]
        depth_files = depth_files[random_numbers: random_numbers + self.sample_n_frames]
        bokeh_files = bokeh_files[random_numbers: random_numbers + self.sample_n_frames]

        # Load frame
        numpy_image = np.array([pil_image_to_numpy(Image.open(os.path.join(video_folder, img))) for img in image_files])

        pixel_values = numpy_to_pt(numpy_image)

        # Load bokeh frames
        f_list = []
        k_list = []
        bokeh_img_list = []
        bokeh_img_list_ori = []
        for i in range(self.sample_n_frames):
            if self.K[idx] == 'rand':
                k_list.append(float(bokeh_files[i].split('_k_')[1].replace('.jpg', '')))
                f_list.append(float(bokeh_files[i].split('zf_')[1].split('_k_')[0]))
            else:
                f_list.append(float(bokeh_files[i].split('zf_')[1].replace('.jpg', '')))
                k_list.append(float(self.K[idx]))
            bokeh_img_list.append(pil_image_to_numpy(Image.open(os.path.join(bokeh_folder, bokeh_files[i]))))
            if self.return_ori_data:
                bokeh_img_list_ori.append(Image.open(os.path.join(bokeh_folder, bokeh_files[i])).resize((self.sample_size[1],self.sample_size[0])))

        bokeh_pixel_values = numpy_to_pt(np.array(bokeh_img_list))

        # Load depth frames
        mpi_mask_list = []
        coc_list = []
        coc_list_ori = []


        for i in range(self.sample_n_frames):
            rand_num = random.random()
            disparity = np.array(Image.open(os.path.join(depth_folder, depth_files[i])).convert("L"))
            disparity = disparity.astype(np.float64)

            if concept_idx not in ['far2near', 'near2far']:
                concept_idx = int(concept_idx)
                mask_path = os.path.join(mask_folder, f'{random_numbers + i}_{concept_idx}.jpg')
                mask_concept = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, mask_concept = cv2.threshold(mask_concept, 128, 1, cv2.THRESH_BINARY)
                mask_names = [f'{random_numbers + i}_{idx}.jpg' for idx in range(self.concept_num) if idx < concept_idx]
                for mask_name in mask_names:
                    mask = cv2.imread(os.path.join(mask_folder, mask_name), cv2.IMREAD_GRAYSCALE)
                    _, binary_mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
                    mask_concept = cv2.subtract(mask_concept, binary_mask)

                if self.aug_depth and rand_num < self.aug_depth_p:

                    if mask_concept.sum() == 0:
                        return None

                    # ElasticTransform
                    disparity = torch.from_numpy(disparity).float()
                    mask_concept = torch.from_numpy(mask_concept).float()
                    displacement = self.elastic.get_params(alpha=[50.0, 50.0], sigma=[4.0, 4.0], size=list(disparity.shape))
                    disparity_elastic = F.elastic_transform(disparity.unsqueeze(0), displacement, interpolation=F.InterpolationMode.BILINEAR, fill=0)[0]
                    mask_elastic = F.elastic_transform(mask_concept.unsqueeze(0), displacement, interpolation=F.InterpolationMode.NEAREST, fill=0)[0]
                    disparity = disparity * (1 - mask_elastic) + disparity_elastic * mask_elastic
                    disparity = torch.clip(disparity, 0, 255)

                    # perlin_noise
                    perlin = self.generate_multi_perlin_noise(scales=[50, 20, 10], amplitudes=[0.5, 0.3, 0.2], mask=(mask_elastic - mask_concept).numpy())
                    perlin = perlin * f_list[i] * 255 * 0.2 - f_list[i] * 255 * 0.1 # 调整范围
                    disparity += perlin

                    # dilate or erode
                    disparity = self.jagged_edges(disparity)
            
            if not isinstance(disparity, torch.Tensor):
                disparity = torch.from_numpy(disparity).float()

            coc_r = torch.abs(disparity - f_list[i] * 255) / 255 # norm

            mpi_mask = gen_mpi_mask(disparity/255, f_list[i], alpha=1.0)
            mpi_mask_list.append(mpi_mask)

            coc_cond = coc_r.repeat(3, 1, 1)
            coc_list.append(coc_cond)

            if self.return_ori_data:
                coc_list_ori.append(Image.fromarray((coc_r * 255).int().cpu().numpy()))
        
        coc_pixel_values = torch.stack(coc_list)
        mpi_mask = torch.stack(mpi_mask_list)  # [4*f, h, w]  true参与计算
        # Load motion values
        motion_values = 127

        if self.return_ori_data:
            pixel_values = self.pixel_transforms(pixel_values)
            bokeh_pixel_values = self.pixel_transforms(bokeh_pixel_values)
            coc_pixel_values = self.pixel_transforms(coc_pixel_values)
            mpi_mask = F.resize(mpi_mask, self.sample_size)

            return pixel_values, bokeh_pixel_values, coc_pixel_values, motion_values, bokeh_img_list_ori, coc_list_ori, torch.asarray(k_list), mpi_mask
        
        else:
            flip_flag = rand_num > 0.5
            if flip_flag:
                pixel_values = F.hflip(pixel_values)
                bokeh_pixel_values = F.hflip(bokeh_pixel_values)
                coc_pixel_values = F.hflip(coc_pixel_values)
                mpi_mask = F.hflip(mpi_mask)

            pixel_values = self.pixel_transforms(pixel_values)
            bokeh_pixel_values = self.pixel_transforms(bokeh_pixel_values)
            coc_pixel_values = self.pixel_transforms(coc_pixel_values)
            mpi_mask = F.resize(mpi_mask, self.sample_size)

            return pixel_values, bokeh_pixel_values, coc_pixel_values, motion_values, torch.asarray(k_list), mpi_mask
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_result = self.get_batch(idx) 
        if batch_result is not None:
            if self.return_ori_data:
                pixel_values, bokeh_pixel_values, coc_pixel_values, motion_values, bokeh_img_list_ori, coc_list_ori, k, mpi_mask = batch_result
                sample = dict(pixel_values=pixel_values, bokeh_pixel_values=bokeh_pixel_values, coc=coc_pixel_values, motion_values=motion_values, gt_imgs=bokeh_img_list_ori, coc_imgs=coc_list_ori, k=k, mpi_mask=mpi_mask)
            else:
                pixel_values, bokeh_pixel_values, coc_pixel_values, motion_values, k, mpi_mask = batch_result
                sample = dict(pixel_values=pixel_values, bokeh_pixel_values=bokeh_pixel_values, coc=coc_pixel_values, motion_values=motion_values, k=k, mpi_mask=mpi_mask)
            return sample
        else:
            idx = random.randint(0, self.length - 1)
            return self.__getitem__(idx)
    

def collate_fn_val_coc_mpi_whole_video(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch[0]])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    coc_pixel_values = torch.stack([example["coc"] for example in batch[0]])
    coc_pixel_values = coc_pixel_values.to(memory_format=torch.contiguous_format).float()
    motion_values = [example["motion_values"] for example in batch[0]]
    k = torch.stack([example["k"] for example in batch[0]])
    mpi_mask = torch.stack([example["mpi_mask"] for example in batch[0]])
    weight = torch.stack([example["weight"] for example in batch[0]])
    return {"pixel_values": pixel_values, 'coc':coc_pixel_values, "motion_values": motion_values,'k':k, 'mpi_mask':mpi_mask, 'weight':weight}