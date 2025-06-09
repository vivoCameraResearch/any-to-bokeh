import argparse
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm

from video_depth_anything.video_depth import VideoDepthAnything

def load_images_as_batch(image_folder, channels=3):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if len(image_files) == 0:
        raise ValueError("No image files were found!")

    batch = []
    for image_name in image_files:
        img_path = os.path.join(image_folder, image_name)
        
        img = cv2.imread(img_path)  # BGR
        if img is None:
            print(f"Skip unreadable images: {img_path}")
            continue

        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        batch.append(img)

    batch_array = np.stack(batch, axis=0)  # (batch_size, height, width, channels)
    
    return batch_array

def img_synthsis(class_names, img_folder, mask_folder, disp_dir):
    device = torch.device("cuda")
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    video_depth_anything = VideoDepthAnything(**model_configs['vitl'])
    video_depth_anything.load_state_dict(torch.load('./checkpoints/video_depth_anything_vitl.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(device)
    video_depth_anything.eval()
    
    for class_name in tqdm(class_names, total=len(class_names)):
        cls_mask_folder = os.path.join(mask_folder, class_name)
        image_folder = os.path.join(img_folder, class_name)
        mask_name = sorted(os.listdir(cls_mask_folder))

        os.makedirs(os.path.join(disp_dir, class_name), exist_ok=True)

        frames = load_images_as_batch(image_folder, channels=3)
        depths, _ = video_depth_anything.infer_video_depth(frames, 1, input_size=518, device=device, fp32=False)
        
        disparitys = depths
        d_min, d_max = disparitys.min(), disparitys.max()

        for i in range(len(disparitys)):
            disp = disparitys[i]
            mask_path = os.path.join(cls_mask_folder, mask_name[i])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
            binary_mask = torch.from_numpy(binary_mask).bool()

            disp_norm = ((disp - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            disp_concept = disp_norm[binary_mask].mean() / 255
            if np.isnan(disp_concept):
                continue
            cv2.imwrite(os.path.join(disp_dir, class_name, mask_name[i].replace('.png', '_zf_{:.5f}.png'.format(disp_concept))), disp_norm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything Processing')
    
    parser.add_argument('--img_folder', type=str,
                        default='demo_dataset/videos',
                        help='Path to folder containing input images organized by class')
    
    parser.add_argument('--mask_folder', type=str,
                        default='demo_dataset/mask',
                        help='Path to folder containing masks organized by class (default: demo_dataset/mask)')
    
    parser.add_argument('--disp_dir', type=str,
                        default='demo_dataset/disp_test',
                        help='Output directory for disparity maps (default: demo_dataset/disp_test)')

    args = parser.parse_args()

    # Validate required arguments
    if not args.img_folder:
        raise ValueError("--img_folder argument is required")

    # Get class names from image folder structure
    try:
        class_names = os.listdir(args.img_folder)
        print(f"Found {len(class_names)} classes to process")
        
        # Process each class
        img_synthsis(
            class_names=class_names,
            img_folder=args.img_folder,
            mask_folder=args.mask_folder,
            disp_dir=args.disp_dir
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
