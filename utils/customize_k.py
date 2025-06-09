import os
import shutil

ori_disp_path = 'demo_dataset/disp/breakdance'
target_disp_path = 'demo_dataset/disp_change_k/breakdance'

# Create target directory if it doesn't exist
os.makedirs(target_disp_path, exist_ok=True)

def sort_frames(frame_name):
    return int(frame_name.split('_')[0].split('.')[0])
# Get all files in source directory
files = [f for f in os.listdir(ori_disp_path)]

files = sorted(files, key=sort_frames)

k = [0] * len(files)
power = 2
k = [int(1 + (32 - 1) * ((i / (len(k) - 1)) ** power)) for i in range(len(k))][::-1]

for i, file in enumerate(files):
    new_name = file.split('.png')[0] + '_k_{}'.format(str(k[i])) + '.png'  # Define the k for each frame
    
    src_path = os.path.join(ori_disp_path, file)
    dst_path = os.path.join(target_disp_path, new_name)
    
    shutil.copy2(src_path, dst_path)

print(f"Copied and renamed {len(files)} files from {ori_disp_path} to {target_disp_path}")
