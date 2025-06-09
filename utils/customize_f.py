import os
import shutil
import torch

def create_adaptive_focus_array(image_files, focus_values):
    """
    :param image_files: Video frame file list
    :param focus_values: 3 focus value, e.g., [focus1, focus2, focus3]
    :return: Generated array of focus values (normalized to [0,1])
    """
    n = len(image_files)
    focus1, focus2, focus3 = focus_values
    
    TRANSITION_RATIO = 0.7  # Transition interval 30% of total length
    
    total_transition_len = int(n * TRANSITION_RATIO)
    total_focus_len = n - total_transition_len
    
    transition_lens = [
        max(1, int(total_transition_len / 2)) for _ in range(2)
    ]
    
    focus_lens = [
        max(1, int(total_focus_len / 3)) for _ in range(3)
    ]
    f = torch.zeros(n)
    ptr = 0
    
    # --- Focus1 ---
    end = ptr + focus_lens[0]
    f[ptr:end] = torch.ones(focus_lens[0]) * (focus1 / 255)
    
    # --- Transition1 (focus1 -> focus2) ---
    ptr = end
    end = ptr + transition_lens[0]
    f[ptr:end] = torch.linspace(focus1/255, focus2/255, transition_lens[0])
    
    # --- Focus2 ---
    ptr = end
    end = ptr + focus_lens[1]
    f[ptr:end] = torch.ones(focus_lens[1]) * (focus2 / 255)
    
    # --- Transition2 (focus2 -> focus3) ---
    ptr = end
    end = ptr + transition_lens[1]
    f[ptr:end] = torch.linspace(focus2/255, focus3/255, transition_lens[1])

    # --- Focus3 ---
    ptr = end
    f[ptr:] = focus3 / 255
    
    return f


ori_disp_path = 'demo_dataset/disp/_Dh3EhO6vbY_69'
target_disp_path = 'demo_dataset/disp_change_f/_Dh3EhO6vbY_69'

# Create target directory if it doesn't exist
os.makedirs(target_disp_path, exist_ok=True)

def sort_frames(frame_name):
    return int(frame_name.split('_')[0].split('.')[0])
# Get all files in source directory
files = [f for f in os.listdir(ori_disp_path)]

files = sorted(files, key=sort_frames)

focus1 = 234
focus2 = 139
focus3 = 31
f = create_adaptive_focus_array(files, ([focus1, focus2, focus3]))

for i, file in enumerate(files):
    new_name = file.split('.jpg')[0] + '_zf_{:5f}'.format(f[i].item()) + '.png'  # Define the f for each frame
    
    src_path = os.path.join(ori_disp_path, file)
    dst_path = os.path.join(target_disp_path, new_name)
    
    shutil.copy2(src_path, dst_path)

print(f"Copied and renamed {len(files)} files from {ori_disp_path} to {target_disp_path}")
