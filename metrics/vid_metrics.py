import os

import torch
from einops import rearrange
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from metrics.utils import read_video_frames, scan_files_in_dir
from metrics.fid_metrics.vfid import vfid
from huggingface_hub import snapshot_download
import ipdb
import multiprocessing as mp

class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
    
    def extract_id_from_filename(self, filename):
        # find first number in filename
        id = filename.split('_')[0]
        return id
    
    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={'.mp4'})
        
        pred_files = scan_files_in_dir(self.pred_folder, postfix={'.mp4'})
        pred_ids = [self.extract_id_from_filename(pred_file.name) for pred_file in pred_files]
        gt_files = [file for file in gt_files if self.extract_id_from_filename(file.name) in pred_ids]
        
        gt_paths = [file.path for file in gt_files]
        gt_names = [file.name for file in gt_files]
        pred_paths = [file.path for file in pred_files if file.name in gt_names]
        gt_paths.sort()
        pred_paths.sort()
        return list(zip(gt_paths, pred_paths))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]
        gt = read_video_frames(gt_path, normalize=False).squeeze(0)
        pred = read_video_frames(pred_path, normalize=False).squeeze(0)

        # crop to same long side
        min_len = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :min_len]
        pred = pred[:, :min_len]
        
        # 确保数据类型一致
        gt = gt.float()
        pred = pred.float()
        
        return gt, pred


def copy_resize_gt(gt_folder, height):
    new_folder = f"{gt_folder}_{height}"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        w, h = img.size
        new_w = int(w * height / h)
        img = img.resize((new_w, height), Image.LANCZOS)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    num_frames = 0
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        ssim_score += ssim(pred, gt) * batch_size
        num_frames += batch_size
    return ssim_score / num_frames

@torch.no_grad()
def psnr(dataloader):
    num_frames = 0
    psnr_score = 0
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating PSNR"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        psnr_score += psnr_metric(pred, gt) * batch_size
        num_frames += batch_size
    return psnr_score / num_frames

@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to("cuda")
    score = 0
    num_frames = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
        num_frames += batch_size
    return score / num_frames

@torch.no_grad()
def temporal_consistency(dataloader):
    frame_metrics = []
    for gt, pred in tqdm(dataloader, desc="Calculating temporal_consistency"):
        num_frames = gt.size(1)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        
        for i in range(num_frames - 1):
            # 计算相邻帧的差异
            pred_diff = pred[i+1] - pred[i]
            target_diff = gt[i+1] - gt[i]
            # 计算差异之间的L1距离
            metric = torch.mean(torch.abs(pred_diff - target_diff)).item()
            frame_metrics.append(metric)
        # 计算平均指标
        avg_metric = sum(frame_metrics) / len(frame_metrics)

    return avg_metric

def compute_fvd(gt_folder, pred_folder):
    # Compute the FVD on two sets of videos.
    from cdfvd import fvd

    # 测试FVD之前一定要将之前gt_folder和pred_folder中的cache删去，不然可能会有问题
    evaluator = fvd.cdfvd('videomae', ckpt_path="./vit_g_hybrid_pt_1200e_ssv2_ft.pth")
    evaluator.compute_real_stats(evaluator.load_videos(gt_folder, data_type='video_folder', sequence_length=25, batch_size=16))
    evaluator.compute_fake_stats(evaluator.load_videos(pred_folder, data_type='video_folder', sequence_length=25, batch_size=16))
    score = evaluator.compute_fvd_from_stats()
    return score
    
def compute_flow_error(dataloader):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).eval().cuda()
    preprocess = weights.transforms()

    frame_metrics = []
    for gt, pred in tqdm(dataloader, desc="Calculating temporal_consistency"):
        # (1, C, F, H, W)
        curr_vid_gt = gt[0].permute(1,0,2,3).cuda()  # (F, C, H, W)
        curr_vid_pred = pred[0].permute(1,0,2,3).cuda()  # (F, C, H, W)

        for i in range(1, curr_vid_gt.shape[0], 1):
            curr_frame_gt = curr_vid_gt[i: i+1]
            curr_frame_pred = curr_vid_pred[i: i+1]

            prev_frame_gt = curr_vid_gt[i-1: i]
            prev_frame_pred = curr_vid_pred[i-1: i]
            
            # 预处理
            frame1_gt, frame2_gt = preprocess(prev_frame_gt, curr_frame_gt)

            frame1_pred, frame2_pred = preprocess(prev_frame_pred, curr_frame_pred)

            with torch.no_grad():
                flow_gt = model(frame1_gt, frame2_gt)[-1]
                flow_pred = model(frame1_pred, frame2_pred)[-1]
            
            # 计算光流变换误差
            flow_error = torch.mean(torch.abs(flow_gt - flow_pred)).item()
            frame_metrics.append(flow_error)

            # 保存光流图
            # flow_rgb = flow_to_rgb(flow_up)
    avg_metric = sum(frame_metrics) / len(frame_metrics)
    return avg_metric

def eval(vfid_ckpt_path, num_workers, gt_folder, pred_folder, save_path):
    os.makedirs(save_path, exist_ok=True)
    # Calculate Metrics
    header = []
    row = []

    header += ["FVD"]
    fvd_score = compute_fvd(gt_folder, pred_folder)
    row += [fvd_score]

    # Form dataset
    dataset = EvalDataset(gt_folder, pred_folder)
    dataloader = torch.utils.data.DataLoader(  # batch size 只能是1
        dataset, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False
    )

    header += ["flow_err"]
    fe = compute_flow_error(dataloader)
    row += [fe]

    header += ["temporal_consistency", "VFID_I3D", "VFID_RESNEXT"]
    tc = temporal_consistency(dataloader)
    row += [tc]

    repo_path = snapshot_download(repo_id=vfid_ckpt_path)
    vfid_resnext = vfid(gt_folder, pred_folder, ckpt_path=repo_path)
    row += [vfid_resnext["i3d"], vfid_resnext["resnext"]]
    
    
    header += ["SSIM", "LPIPS", "PSNR"]
    ssim_ = ssim(dataloader).item()
    lpips_ = lpips(dataloader).item()
    psnr_ = psnr(dataloader).item()
    row += [ssim_, lpips_, psnr_]
    
    # Print Results
    print("GT Folder  : ", gt_folder)
    print("Pred Folder: ", pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)
    print(table)
    # 保存为 txt
    output_file = os.path.join(save_path, "{}.txt".format(pred_folder.split('/')[-2]))
    with open(output_file, "w") as f:
        f.write(str(table))  # str(table) 会打印出文本表格样式

         
if __name__ == "__main__":
    vfid_ckpt_path = 'zhengchong/VFID'
    num_workers = 4
    gt_folder = './z_vis/z_vis_f5_1024_kemb_spa_coc_add_data_50000_test_dataset/gt'
    save_path = './z_rebuttal_metric'

    ##### 推理单个文件夹的指标 #####
    pred_folder = './z_rebuttal/f4_1024_kemb_spa_coc_kinj_add_data_50k_alpha1.0_stage2_temp_Elastic+edge_perlin+dilate_erode_20000_30000_vae_test_datasetdepth+elastic+gaussian_whole_vid/pred'
    os.makedirs(pred_folder, exist_ok=True)
    eval(vfid_ckpt_path, num_workers, gt_folder, pred_folder, save_path)
    ##### 推理单个文件夹的指标 #####

    # ##### 推理root内部所有文件夹的指标 #####
    # pred_root = 'z_rebuttal'
    # gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 可用 GPU id
    # def worker(gpu_id, tasks):
    #     # 仅暴露一张 GPU 给该进程
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #     try:
    #         torch.cuda.set_device(0)  # 进程内只有一张卡，因此设为0
    #     except Exception:
    #         pass

    #     for pred_folder in tasks:
    #         print(f'GPU {gpu_id} -> {pred_folder}')
    #         eval(vfid_ckpt_path, num_workers, gt_folder, pred_folder, save_path)

    # folders = [os.path.join(pred_root, d, 'pred') for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
    # buckets = [[] for _ in gpus]
    # for i, f in enumerate(folders):
    #     buckets[i % len(gpus)].append(f)
    # procs = []
    # for gpu_id, tasks in zip(gpus, buckets):
    #     if not tasks:
    #         continue
    #     p = mp.Process(target=worker, args=(gpu_id, tasks))
    #     p.start()
    #     procs.append(p)

    # for p in procs:
    #     p.join()

    # for i in os.listdir(pred_root):
    #     os.makedirs(pred_folder, exist_ok=True)
    #     pred_folder = os.path.join(pred_root, i)
    #     eval(vfid_ckpt_path, num_workers, gt_folder, pred_folder, save_path)
    # ##### 推理root内部所有文件夹的指标 #####