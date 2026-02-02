import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

def edge_map(img):
    """提取灰度图的边缘幅值"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)
    return magnitude

def correlation(a, b, mask=None):
    """计算相关系数（可加mask）"""
    if mask is not None:
        a = a[mask > 0]
        b = b[mask > 0]
    if a.size == 0 or b.size == 0:
        return 0
    a_mean, b_mean = np.mean(a), np.mean(b)
    numerator = np.sum((a - a_mean) * (b - b_mean))
    denominator = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    return numerator / denominator if denominator != 0 else 0

def calc_smoothness(epi_fg_list):
    """计算前景边缘平顺度（0-1，越接近 1 越平顺）"""
    if len(epi_fg_list) < 2:
        return 0
    diffs = np.diff(epi_fg_list)
    std_diff = np.std(diffs)
    max_possible_diff = 1.0
    smoothness_score = 1 - (std_diff / max_possible_diff)
    return max(smoothness_score, 0)

def bokeh_epi_eval_single(ref_frames_dir, bokeh_video_path, mask_dir, save_prefix, save_root):
    """单个视频的 EPI_fg 计算"""
    ref_files = sorted(os.listdir(ref_frames_dir))
    mask_files = sorted(os.listdir(mask_dir))
    cap_bokeh = cv2.VideoCapture(bokeh_video_path)

    total_frames = min(len(ref_files), len(mask_files), int(cap_bokeh.get(cv2.CAP_PROP_FRAME_COUNT)))

    epi_fg_list = []
    smoothness_list = []

    for idx in range(total_frames):
        frame_ref = cv2.imread(os.path.join(ref_frames_dir, ref_files[idx]))
        ret_bokeh, frame_bokeh = cap_bokeh.read()
        frame_mask = cv2.imread(os.path.join(mask_dir, mask_files[idx]), cv2.IMREAD_GRAYSCALE)

        if frame_ref is None or not ret_bokeh or frame_mask is None:
            continue

        target_height, target_width = frame_bokeh.shape[:2]
        frame_ref = cv2.resize(frame_ref, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame_mask = cv2.resize(frame_mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        kernel_size = (5, 5)  # 结构元素的大小
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        # 对 mask 进行膨胀操作
        frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)

        mask_fg = (frame_mask > 127).astype(np.uint8)

        edges_ref = edge_map(frame_ref)
        edges_bokeh = edge_map(frame_bokeh)

        epi_fg = correlation(edges_ref, edges_bokeh, mask_fg)
        epi_fg_list.append(epi_fg)


    cap_bokeh.release()

    smoothness_score = calc_smoothness(epi_fg_list)

    if save_root:
        # 保存结果
        # np.savez(f"{save_prefix}.npz", epi_fg_list=np.array(epi_fg_list), smoothness_score=smoothness_score)
        pd.DataFrame({"EPI_fg": epi_fg_list}).to_csv(os.path.join(save_root, f"{save_prefix}.csv"), index=False)

        # 绘图
        plt.figure()
        plt.plot(epi_fg_list, label="EPI_fg")
        plt.xlabel("Frame Index")
        plt.ylabel("Score")
        plt.legend()
        plt.title(f"EPI_fg - {save_prefix}--{np.mean(epi_fg_list):.4f}")
        plt.savefig(os.path.join(save_root, f"{save_prefix}.png"))
        plt.close()

    # print(f"{save_prefix} - 平均 EPI_fg: {np.mean(epi_fg_list):.4f}, 平顺度: {smoothness_score:.4f}")

    return epi_fg_list, smoothness_score

def evaluate_multiple_videos(video_triplets, exp_name, save_root=None):
    all_epi_fg = []
    all_smoothness = []
    min_len = float("inf")

    for idx, (ref_frames_dir, bokeh_video_path, mask_dir) in tqdm(enumerate(video_triplets), total=len(video_triplets)):
        save_prefix = f"{exp_name}/{idx+1}"
        epi_fg_list, smoothness_score = bokeh_epi_eval_single(ref_frames_dir, bokeh_video_path, mask_dir, save_prefix, save_root)
        all_epi_fg.append(np.mean(epi_fg_list))
        all_smoothness.append(smoothness_score)
        min_len = min(min_len, len(epi_fg_list))
   

    print(f"平均 EPI_fg: {np.mean(all_epi_fg):.4f}, 平均平顺度: {np.mean(all_smoothness):.4f}")

if __name__ == "__main__":

    exp_dict = {
        'bokeme': './vid_bokeh_baseline_refine_no_norm/bokehme_davis/pred',
        'our': './f4_1024_kemb_spa_coc_kinj_add_data_50k_alpha1.0_stage2_temp_Elastic+edge_perlin+dilate_erode_20000_30000_vae_davis_k16_whole_vid/pred',
    }
    
    # 读取csv
    df = pd.read_csv("./csv/davis_k16.csv")
    video_folder = df['aif_folder'].tolist()
    mask_folder = df['mask_folder'].tolist()
    # save_root = None  # 不保存绘图
    save_root = './z_edge'

    for exp_name, bokeh_videos_root in exp_dict.items():
        print(exp_name)
        if save_root:
            os.makedirs(os.path.join(save_root, exp_name), exist_ok=True)
        video_triplets = []
        for i in range(len(video_folder)):
            video_triplets.append((video_folder[i], os.path.join(bokeh_videos_root, f'{i}.mp4'), mask_folder[i]))

        evaluate_multiple_videos(video_triplets, exp_name, save_root)