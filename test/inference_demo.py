import os
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.abspath(__file__).split('test')[0])
from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video
from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
from models.unet import UNetSpatioTemporalConditionModel
import imageio
from tqdm import tqdm
from models.vae import AutoencoderKLTemporalDecoder
import argparse

def create_video(video_list, output_video, fps=30):
    num_frames = len(video_list[0])

    writer = imageio.get_writer(output_video, fps=fps, codec='libx264', quality=8)

    for f in range(num_frames):
        images = []
        for video in video_list:
            if video[f].mode != 'RGB':
                video[f] = video[f].convert('RGB')
            images.append(np.asarray(video[f]))

        frame = np.concatenate(images, axis=1)

        writer.append_data(frame)

    writer.close()

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Depth Anything Processing')

    parser.add_argument('--val_csv_path', type=str, default='csv_file/demo.csv',
                    help='Path to validation CSV file (default: csv_file/demo.csv)')
    parser.add_argument('--unet_path', type=str, default='checkpoints/unet',
                    help='Path to UNET model checkpoint (default: checkpoints/unet)')
    parser.add_argument('--vae_path', type=str, default='checkpoints/vae',
                    help='Path to VAE model checkpoint (default: checkpoints/vae)')

    args = parser.parse_args()


    height = 576
    width = 1024
    group_frames = 8
    overlap_frames = 4
    infer_clip_num = 2  # Only support 2
    val_save_dir = 'output'
    os.makedirs(val_save_dir, exist_ok=True)

    val_dataset = BokehCocMPIDatasetWholeVid(args.val_csv_path, sample_size=(height, width), overlap_frames=overlap_frames, group_frames=group_frames)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=64,
        collate_fn=collate_fn_val_coc_mpi_whole_video
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.unet_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False, local_files_only=True,
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.vae_path,
        local_files_only=True,torch_dtype=torch.float16)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        vae=vae,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16, variant="fp16", local_files_only=True,
    )
    pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    for val_img_idx, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        video = []
        latent_cache = None
        for i in range(0, val_batch["coc"].shape[0], infer_clip_num):
            is_last_batch = i + infer_clip_num >= val_batch["coc"].shape[0]

            weight = val_batch["weight"][i:i+infer_clip_num].to(torch.float16).to('cuda', non_blocking=True)
            validation_control_images = val_batch["coc"][i:i+infer_clip_num].to('cuda', non_blocking=True)
            val_ref_images = val_batch["pixel_values"][i:i+infer_clip_num].to('cuda', non_blocking=True)
            input_k = val_batch["k"][i:i+infer_clip_num].to(torch.float16).to('cuda', non_blocking=True)
            mpi_mask = val_batch["mpi_mask"][i:i+infer_clip_num].to(torch.float16).to('cuda', non_blocking=True)

            video_frames, latent_cache = pipe(
                validation_control_images,
                val_ref_images,
                input_k=input_k,
                mpi_mask=mpi_mask,
                latent_cache=latent_cache,
                is_last_batch=is_last_batch,
                overlap_frame=overlap_frames,
                weight=weight,
                height=height,
                width=width,
                num_frames=group_frames,
                decode_chunk_size=group_frames,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.0,
                num_inference_steps=1,
                generator=torch.Generator(device='cpu').manual_seed(0),
                batch_size=2 if weight.shape[0] == infer_clip_num else 1,
            )
            video.extend(video_frames[0])
        create_video([video], os.path.join(val_save_dir,f"{val_img_idx}.mp4"), fps=20)