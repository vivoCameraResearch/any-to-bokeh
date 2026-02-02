import sys
sys.path.append('/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/diffusers-main/z_bokeh')
sys.path.append('/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/diffusers-main')
import torchvision.transforms as transforms
import json
import os, glob
import numpy as np
from PIL import Image
import math
import copy
from tqdm import tqdm
import torch
import cv2
import shutil
import multiprocessing
from z_bokeh.data_process.bokeh_kernel import ModuleRenderRT
import random
import ipdb
import time


def divide_rectangle_into_grid_and_get_centers(width, height, rows, cols):
    """
    将矩形均匀分成 rows * cols 的网格，并返回每一块的centers
    :param x_min: 矩形左上角的 x 坐标
    :param y_min: 矩形左上角的 y 坐标
    :param x_max: 矩形右下角的 x 坐标
    :param y_max: 矩形右下角的 y 坐标
    :param rows: 行数
    :param cols: 列数
    :return: 每一块的centers列表 [(cx, cy), ...]
    """
    # 计算每个小矩形的宽度和高度
    row_height = height / rows
    col_width = width / cols
    
    centers = []
    lefttops = []
    rightbotton = []
    # 遍历每个小矩形的行和列
    for i in range(rows):
        for j in range(cols):
            # 计算当前小矩形的左上角坐标
            left = j * col_width
            top = i * row_height
            
            right = left + col_width
            bottom = top + row_height
            
            # 计算小矩形的中心点
            cx = int((left + right) / 2)
            cy = int((top + bottom) / 2)
            
            centers.append(np.asarray([cx, cy]))  # w,h
            lefttops.append(np.asarray([left, top]))  # w,h
            rightbotton.append(np.asarray([right, bottom]))  # w,h
    
    return centers, lefttops, rightbotton

def calculate_iou(rect1, rect2):
    """
    计算两个矩形的 IoU (Intersection over Union)
    :param rect1: (cx1, cy1, w1, h1)
    :param rect2: (cx2, cy2, w2, h2)
    :return: IoU值
    """
    # 计算矩形1和矩形2的左上角和右下角坐标
    x_min_1 = rect1[0]
    y_min_1 = rect1[1]
    x_max_1 = rect1[0] + rect1[2]
    y_max_1 = rect1[1] + rect1[3]

    x_min_2 = rect2[0]
    y_min_2 = rect2[1]
    x_max_2 = rect2[0] + rect2[2]
    y_max_2 = rect2[1] + rect2[3]

    # 计算交集区域的边界
    x_min_intersection = max(x_min_1, x_min_2)
    y_min_intersection = max(y_min_1, y_min_2)
    x_max_intersection = min(x_max_1, x_max_2)
    y_max_intersection = min(y_max_1, y_max_2)

    # 如果没有交集，返回IoU为0
    if x_min_intersection >= x_max_intersection or y_min_intersection >= y_max_intersection:
        return 0

    # 交集的面积
    intersection_area = (x_max_intersection - x_min_intersection) * (y_max_intersection - y_min_intersection)

    # 计算矩形1和矩形2的面积
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    # 并集的面积
    union_area = area_1 + area_2 - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou

def check_iou_lower_threshold(rectangles, threshold):
    """
    检查一组矩形之间的IoU是否大于给定阈值
    :param rectangles: 矩形的左上角 (x, y), 长宽 (w, h) 的列表 [(cx1, cy1, w1, h1), (cx2, cy2, w2, h2), ...]
    :param threshold: IoU的阈值
    :return: False 如果有任意两个矩形的 IoU > threshold，否则返回 FaTruelse
    """
    n = len(rectangles)
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(rectangles[i], rectangles[j])
            if iou > threshold:
                # print(f"矩形 {i} 和 矩形 {j} 的 IoU: {iou:.2f} > {threshold}")
                return False
    return True

def modify_alpha_with_mask(clip_img, mask_img, threshold=0):
    """
    修改图像的 Alpha 通道，根据 mask 修改透明度。
    
    :param clip_path: RGBA 图
    :param mask_path: 用于修改 Alpha 通道的 mask 图像
    :param threshold: 设置透明度阈值，mask值大于阈值的像素会保持原透明度，反之会变透明
    :return: 修改后的图像（PIL Image 对象）
    """
    # 将图像转换为 NumPy 数组，便于处理
    clip_array = np.array(clip_img)
    mask_array = np.array(mask_img)
    
    # 将修改后的 Alpha 通道替换回原图
    clip_array[:, :, 3] = mask_array

    # 将修改后的图像转回 PIL 图像
    modified_clip_img = Image.fromarray(clip_array, mode="RGBA")
    
    return modified_clip_img

def img_synthsis(clip_date, rand_bg, rand_fg, obj_num, res_w, res_h, frame_num, check_iou, dof, dof_bg, save_pickel, bg_files, canvas_centers, canvas_lefttops, canvas_rightbotton, save_dir, gpu_id, seed, randk_flag, gen_test_flag, change_focus_flag):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    module = ModuleRenderRT().to(device)
    for cnt, (n_bg, n_fgs) in enumerate(tqdm(zip(rand_bg, rand_fg), total=len(rand_bg))):
        torch.manual_seed(seed + cnt)
        fg_files = [clip_date[str(n_fg)] for n_fg in n_fgs]
        fg_tokens = ''
        for fg_file in fg_files:
            if fg_file['clip_paths'][0].split('/')[-2] == 'fg':  # 图片合成bokeh
                fg_tokens = fg_tokens + '_' + fg_file['clip_paths'][0].split('/')[-1].replace('.png','')
            elif fg_file['clip_paths'][0].split('/')[-2] == 'original':
                fg_tokens = fg_tokens + '_' + fg_file['clip_paths'][0].split('/')[-1].replace('.jpg','')
            else:
                fg_tokens = fg_tokens + '_' + fg_file['clip_paths'][0].split('/')[-2]
        clip_dir = os.path.join(save_dir, f"clip_{n_bg}_{fg_tokens}")
        if os.path.exists(clip_dir):
            continue
        bokeh_dir = os.path.join(clip_dir, "bokeh")
        aif_dir = os.path.join(clip_dir, "aif")
        disp_dir = os.path.join(clip_dir, "disp")
        mask_dir = os.path.join(clip_dir, "mask")
        os.makedirs(bokeh_dir, exist_ok=True)
        os.makedirs(aif_dir, exist_ok=True)
        os.makedirs(disp_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        error_flag = False
        if save_pickel:
            pickle_dir = os.path.join(save_dir, "ckpt")
            os.makedirs(pickle_dir, exist_ok=True)
        while True:  # 确定填充位置以及填充图像
            all_obj_frame = {}
            all_obj_location = {}
            # center 随机扰动(每个case的center不同)
            w_range_l = canvas_lefttops[:, 0] - canvas_centers[:, 0]
            w_range_r = canvas_rightbotton[:, 0] - canvas_centers[:, 0]
            h_range_l = canvas_lefttops[:, 1] - canvas_centers[:, 1]
            h_range_r = canvas_rightbotton[:, 1] - canvas_centers[:, 1]
            center_rand_w = np.random.randint(w_range_l, w_range_r)
            center_rand_h = np.random.randint(h_range_l, h_range_r)
            centers = copy.deepcopy(canvas_centers)
            centers[:, 0] = centers[:, 0] + center_rand_w
            centers[:, 1] = centers[:, 1] + center_rand_h
            for i, fg_file in enumerate(fg_files):
                clip_paths, clip_mask_paths = fg_file['clip_paths'], fg_file['clip_mask_paths']
                center = centers[i]
                # 处理单个视频
                obj_frame = []
                obj_location = []
                for f, (clip_path, clip_mask_path) in enumerate(zip(clip_paths, clip_mask_paths)):
                    clip_img = Image.open(clip_path).convert("RGBA")
                    clip_mask_img = Image.open(clip_mask_path).convert("L")
                        
                    # 去除边界
                    mask_array = np.array(clip_mask_img)
                    non_transparent_mask = mask_array > 128  # 透明部分值为0，非透明部分大于0
                    # 获取图像的边界
                    non_transparent_pixels = np.where(non_transparent_mask)
                    try:  # 去除没有图像的帧
                        min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
                        min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
                    except:
                        error_flag = True
                        break

                    # 剪裁图像和 mask，只保留非透明区域
                    clip_img = clip_img.crop((min_x, min_y, max_x, max_y))
                    clip_mask_img = clip_mask_img.crop((min_x, min_y, max_x, max_y))

                    if f == 0:  # 定义随机缩放+随机旋转
                        # PIL size：W，H
                        scale = min(res_w / clip_img.size[0], res_h / clip_img.size[1])  # 按比例缩放到背景（占满H或W，方便后续处理）
                        # 随机缩放
                        scale = scale * np.random.uniform(0.5, 0.8)  # 根据数量得到缩放比例
                        # 随机旋转
                        rotate_angle = np.random.uniform(-10, 10)
                    
                    # 随机缩放+随机旋转
                    clip_img = clip_img.resize((int(clip_img.width * scale), int(clip_img.height * scale)))
                    if clip_img.width > res_w or clip_img.height > res_h:  # 去除掉异常值
                        error_flag = True
                        break
                    clip_mask_img = clip_mask_img.resize((int(clip_mask_img.width * scale), int(clip_mask_img.height * scale)))
                    clip_img = clip_img.rotate(rotate_angle)
                    clip_mask_img = clip_mask_img.rotate(rotate_angle)

                    # 重新注入mask
                    clip_img = modify_alpha_with_mask(clip_img,clip_mask_img)
                    
                    # 随机扰动中心点(一个case中单一视频的抖动)
                    w, h = clip_img.size
                    video_center_rand = np.random.randint(-40, 40, 2)
                    center += video_center_rand

                    # 防止出界
                    x_min = int(center[0] - w / 2)
                    y_min = int(center[1] - h / 2)
                    x_max = x_min + w
                    y_max = y_min + h
                    if x_min < 0:
                        x_min = 0
                        x_max = w
                        # print('x min')
                    if y_min < 0:
                        y_min = 0
                        y_max = h
                        # print('y min')
                    if x_max > res_w:
                        x_min = res_w - w
                        x_max = res_w
                        # print('x max')
                    if y_max > res_h:
                        y_min = res_h - h
                        y_max = res_h
                        # print('y max')
                    obj_location.append([x_min, y_min, w, h])
                    obj_frame.append(clip_img)
                
                all_obj_frame.update({i: obj_frame})
                all_obj_location.update({i: obj_location})
            
            if not error_flag:
                check_flag = True
                for key, value in all_obj_location.items():
                    frame_loaction = []
                    if check_flag:
                        for i in range(frame_num):
                            frame_loaction.append(value[i])
                            if not check_iou_lower_threshold(frame_loaction, check_iou):
                                check_flag = False
                                break
                if check_flag:
                    break
            else:
                break
        
        if error_flag:
            shutil.rmtree(os.path.join(clip_dir))
            continue
        # 初始化，背景放在0
        images = torch.zeros((frame_num, obj_num+1, 3, res_h, res_w))  # batch, num_object, C, H, W
        alphas = torch.zeros((frame_num, obj_num+1, 1, res_h, res_w)) # batch, num_object, C, H, W
        coffs = torch.zeros((frame_num, obj_num+1, 3))  # batch, num_object, num_param (a, b, c) (refer to Eq.7 of the paper)
        coffs_input = torch.zeros((frame_num, obj_num+1, 3))  # batch, num_object, num_param (a, b, c) (refer to Eq.7 of the paper)
        # ! fix修改这里，多种深度
        coffs[..., 0:2] = 0
        # 防止超出各自的界限（不然aif渲染有问题，必须保持前后一致性）   TODO：这里可以增加更多的随机性，aif也要跟着修改
        rand_coffs = torch.linspace(dof[0], dof[1], obj_num) + ((dof[1] - dof[0]) / obj_num / 2 * torch.rand((frame_num, obj_num)) - (dof[1] - dof[0]) / obj_num / 4)
        rand_obj = torch.randperm(obj_num)  # 随机更换多个物体的前后顺序
        idx = torch.argsort(rand_obj)
        # 用于渲染aif和disp，随机打乱前后顺序
        coffs[..., 2][..., :obj_num] = rand_coffs[:, rand_obj]
        coffs[..., 2][..., :obj_num] = torch.clip(coffs[..., 2][..., :obj_num], dof[0], dof[1])
        coffs[..., 2][..., -1] = (dof_bg[1] - dof_bg[0]) * torch.rand(frame_num) + dof_bg[0]

        # 用于渲染bokeh，从浅到深排序，和后续将img
        coffs_input[..., 2][..., :obj_num] = rand_coffs
        coffs_input[..., 2][..., :obj_num] = torch.clip(coffs_input[..., 2][..., :obj_num], dof[0], dof[1])
        coffs_input[..., 2][..., -1] = coffs[..., 2][..., -1].clone()

        try:
            bg_img = Image.open(bg_files[n_bg]).convert("RGBA").resize((res_w, res_h))
        except:
            continue
        bg_img = torch.from_numpy(np.array(bg_img)).permute(2, 0, 1).to(device)
        grid_y, grid_x = torch.meshgrid(torch.arange(res_h), torch.arange(res_w), indexing="ij")
        grid_y = grid_y[None, None].to(device)
        grid_x = grid_x[None, None].to(device)
        images = images.to(device)
        alphas = alphas.to(device)
        coffs = coffs.to(device)
        coffs_input = coffs_input.to(device)
        # 添加img
        zf_list = [[] for _ in range(obj_num)]
        for f in range(frame_num):
            aif = torch.zeros((1, 3, res_h, res_w)).to(device)
            disp = torch.zeros((1, 1, res_h, res_w)).to(device)
            # 背景初始化
            images[f, -1, :, :, :] = bg_img[:3,:,:] / 255.0
            alphas[f, -1, :, :, :] = 1
            aif[0] = bg_img[:3,:,:] / 255.0
            disp_i = (1 - coffs[f, -1, 0] * grid_x - coffs[f, -1, 1] * grid_y) / coffs[f, -1, 2]
            disp = disp * (1 - alphas[f, -1]) + disp_i * alphas[f, -1]
            # 将matting图像按照前后关系，注入到images和alphas tensor中
            for obj_n in range(obj_num - 1 , -1 , -1):
                obj_i = idx[obj_n].item()  # 找出对应第i个深度物体的idx
                obj_crop = torch.from_numpy(np.array(all_obj_frame[obj_i][f])).permute(2, 0, 1).to(device)
                x, y, w, h = all_obj_location[obj_i][f]
                images[f, obj_n, :, y:y+h, x:x+w] = obj_crop[:3,:,:]  / 255.0
                alphas[f, obj_n, :, y:y+h, x:x+w] = obj_crop[3,:,:]  / 255.0
            # 渲染aif和disp
            for obj_n in range(obj_num - 1 , -1 , -1):
                obj_i = idx[obj_n].item()  # 找出对应第i个深度物体的idx
                aif = aif * (1 - alphas[f, obj_n]) + images[f, obj_n] * alphas[f, obj_n]
                # 计算深度时转换matting mask
                binary_mask = torch.where(alphas[f, obj_n] > 0.5, 1, 0)
                cv2.imwrite(os.path.join(mask_dir, f'{f}_{obj_n}.jpg'), binary_mask[0].cpu().numpy() * 255)   # 存mask，按照深度存,从深到浅
                disp_i = (1 - coffs[f, obj_i, 0] * grid_x - coffs[f, obj_i, 1] * grid_y) / coffs[f, obj_i, 2]
                disp = disp * (1 - binary_mask) + disp_i * binary_mask
                # zf_list[obj_i].append(torch.mean(disp_i[0][binary_mask.bool()]).item())
                zf_list[obj_n].append(torch.mean(disp_i[0][binary_mask.bool()]).item())

            aif = aif[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255
            aif = cv2.cvtColor(aif, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(aif_dir, f'{f}.jpg'), aif)
            disp = disp[0][0].detach().clone().cpu().numpy() * 255
            cv2.imwrite(os.path.join(disp_dir, f'{f}.jpg'), disp)
        if save_pickel:
            torch.save([images, alphas, coffs_input], os.path.join(save_dir, 'ckpt', str(cnt) + '.pth'))  # 这三者要一一对应
        
        # generate bokeh
        if randk_flag or gen_test_flag:
            k_dir = os.path.join(bokeh_dir, "k=rand")
            # randk_mod = torch.randint(0, 3, (1,))  # 2 随机k，1正序k，0倒序k
            randk_mod = torch.randint(0, 2, (1,))  # 2 随机k，1正序k，0倒序k
            if randk_flag:
                if randk_mod == 2:
                    k = torch.randint(1, 32, (25,)).to(device)  # 不同帧不同k
                else:
                    start = torch.randint(1, 32 - 25 + 1, (1,)).item()
                    # 生成从start开始的25个连续数字
                    k = torch.arange(start, start + 25).to(device) 
                    if randk_mod == 1:
                        k = k.flip(0)

            if gen_test_flag:
                # 生成eval时用
                arr = torch.tensor([32,16,8,4])
                idx = torch.randint(0, len(arr), (obj_num,))
                k = arr[idx].to(device)

            for n, zf in enumerate(zf_list):
                bokeh_save_dir = os.path.join(k_dir, "obj_" + str(n))
                os.makedirs(bokeh_save_dir, exist_ok=True)
                zf = torch.tensor(zf).to(device)
                bokeh = module(images, alphas, coffs_input, k[n].repeat(frame_num), 1 / zf, 101)
                torch.cuda.synchronize(device)
                for f in range(frame_num):
                    # bokeh_b1 = module(images[f:f+1], alphas[f:f+1], coffs[f:f+1], k, 1 / zf[f:f+1], 101)
                    # torch.cuda.synchronize(device)
                    # cv2.imwrite(os.path.join(bokeh_save_dir, 'b1_{:d}_zf_{:.5f}.jpg'.format(f, zf[f].item())), bokeh_b1[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255)
                    bokeh_f = bokeh[f].detach().clone().permute(1, 2, 0).cpu().numpy() * 255
                    bokeh_f = cv2.cvtColor(bokeh_f, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(bokeh_save_dir, '{:d}_zf_{:.5f}_k_{:d}.jpg'.format(f, zf[f].item(), k[n].item())), bokeh_f)

        else:
            k_option = torch.tensor([28,24,20,16,12])
            rand_index = torch.randint(0, len(k_option), (1,)).item()
            k = k_option[rand_index]
            k_dir = os.path.join(bokeh_dir, "k=" + str(k.item()))

            if change_focus_flag:
                random_value = torch.rand(1).item()
                if random_value <= 0.5:
                    zf = torch.linspace(zf_list[-1][0], zf_list[0][-1], frame_num).to(device)  # 远到近
                    bokeh_save_dir = os.path.join(k_dir, "obj_" + 'far2near')
                else:
                    zf = torch.linspace(zf_list[0][0], zf_list[-1][-1], frame_num).to(device)  # 近到远
                    bokeh_save_dir = os.path.join(k_dir, "obj_" + 'near2far')

                os.makedirs(bokeh_save_dir, exist_ok=True)
                bokeh = module(images, alphas, coffs_input, k.item(), 1 / zf, 101)
                torch.cuda.synchronize(device)
                for f in range(frame_num):
                    # bokeh_b1 = module(images[f:f+1], alphas[f:f+1], coffs[f:f+1], k, 1 / zf[f:f+1], 101)
                    # torch.cuda.synchronize(device)
                    # cv2.imwrite(os.path.join(bokeh_save_dir, 'b1_{:d}_zf_{:.5f}.jpg'.format(f, zf[f].item())), bokeh_b1[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255)
                    bokeh_f = bokeh[f].detach().clone().permute(1, 2, 0).cpu().numpy() * 255
                    bokeh_f = cv2.cvtColor(bokeh_f, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(bokeh_save_dir, '{:d}_zf_{:.5f}.jpg'.format(f, zf[f].item())), bokeh_f)

            # for k in [32,16,8,4]:
            # # for k in [16]:
            #     k_dir = os.path.join(bokeh_dir, "k=" + str(k))
            #     for n, zf in enumerate(zf_list):
            #         bokeh_save_dir = os.path.join(k_dir, "obj_" + str(n))
            #         os.makedirs(bokeh_save_dir, exist_ok=True)
            #         zf = torch.tensor(zf).to(device)
            #         bokeh = module(images, alphas, coffs_input, k, 1 / zf, 101)
            #         torch.cuda.synchronize(device)
            #         for f in range(frame_num):
            #             # bokeh_b1 = module(images[f:f+1], alphas[f:f+1], coffs[f:f+1], k, 1 / zf[f:f+1], 101)
            #             # torch.cuda.synchronize(device)
            #             # cv2.imwrite(os.path.join(bokeh_save_dir, 'b1_{:d}_zf_{:.5f}.jpg'.format(f, zf[f].item())), bokeh_b1[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255)
            #             bokeh_f = bokeh[f].detach().clone().permute(1, 2, 0).cpu().numpy() * 255
            #             bokeh_f = cv2.cvtColor(bokeh_f, cv2.COLOR_BGR2RGB)
            #             cv2.imwrite(os.path.join(bokeh_save_dir, '{:d}_zf_{:.5f}.jpg'.format(f, zf[f].item())), bokeh_f)
            #         # ipdb.set_trace()

if __name__ == '__main__':
    # back_files_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/Backgrounds"
    # back_files_path2 = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/BG_imgs"
    back_files_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/BG_test"  # 背景路径
    # clip_json = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/test/clip/clip_f25.json"
    # clip_json = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/merge_f25.json"
    clip_json = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/merge_f25_test.json"
    
    # save_dir = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/video_bokeh_change_f"
    # save_dir = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/video_bokeh_change_k"
    save_dir = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/video_bokeh_test2"

    clip_data = json.load(open(clip_json, 'r'))


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    synthsis_images_num = 300  # (280 k=1 4h  h20)   (280 k=4 10h h800)  # 生成数量
    obj_num = 4  # 前景数量
    # obj_num = 9  # 前景数量
    res_w, res_h = 1024, 576
    frame_num = 25  # 帧数
    check_iou = 1.0  # iou检查
    dof = (1, 5)  # 深度范围 disp要倒过来
    dof_bg = (100, 200)  # 背景深度范围
    save_pickel = False
    bg_files = sorted(glob.glob(os.path.join(back_files_path, "*.jpg")))
    # bg_files2 = sorted(glob.glob(os.path.join(back_files_path2, "*.jpg")))
    # bg_files.extend(bg_files2)
    randk_flag = False  # 用于生成一个视频中不同k的bokeh,生成test数据集也用这个
    gen_test_flag = True
    change_focus_flag = False
    assert not (randk_flag and gen_test_flag), '不能同时为 ture'
    np.random.seed(20)

    rand_bg = np.random.randint(0, len(bg_files), size=synthsis_images_num)
    rand_fg = np.random.randint(0, len(clip_data), size=(synthsis_images_num, obj_num))
    canvas_centers, canvas_lefttops, canvas_rightbotton = divide_rectangle_into_grid_and_get_centers(res_w - 20, res_h - 20, math.ceil(math.sqrt(obj_num)), math.floor(math.sqrt(obj_num)))
    canvas_centers = np.vstack(canvas_centers)
    canvas_lefttops = np.vstack(canvas_lefttops)
    canvas_rightbotton = np.vstack(canvas_rightbotton)
    num_gpus = torch.cuda.device_count()

    num_threads = 5  # 可以根据你的系统和任务调整线程数
    # 计算每个线程应该处理的图像组合数量
    combinations_per_thread = synthsis_images_num // num_threads

    processes = []
    for i in range(num_threads):
        gpu_id = i % num_gpus
        start_index = i * combinations_per_thread
        # 确保最后一个线程处理所有剩余的组合
        end_index = (i + 1) * combinations_per_thread if i != num_threads - 1 else synthsis_images_num
        args = (copy.deepcopy(clip_data), rand_bg[start_index:end_index], rand_fg[start_index:end_index], obj_num, res_w, res_h, frame_num, check_iou, dof, dof_bg, save_pickel, bg_files, canvas_centers, canvas_lefttops, canvas_rightbotton, save_dir, gpu_id, rand_bg[0], randk_flag, gen_test_flag, change_focus_flag)
        # img_synthsis(*args)  # 调试
        p = multiprocessing.Process(target=img_synthsis, args=args, daemon=True)
        processes.append(p)
        p.start()
    # 等待所有进程完成
    for p in processes:
        p.join()


'''
上述的json样式：merge_f25_test.json，下面这是图片的格式
{
    "0": {
        "clip_paths": [
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/fg/m_6be54a7b.png"
        ],
        "clip_mask_paths": [
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png",
            "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11167024/dataset/matting/AM-2k/train/mask/m_6be54a7b.png"
        ]
    },
}


后面是25帧视频matting的格式
{
    "0": {
        "clip_paths": [
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00000.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00004.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00008.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00012.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00016.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00020.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00024.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00028.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00032.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00036.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00040.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00044.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00048.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00052.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00056.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00060.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00064.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00068.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00072.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00076.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00080.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00084.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00088.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00092.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/fgr/0000/00096.jpg"
        ],
        "clip_mask_paths": [
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00000.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00004.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00008.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00012.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00016.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00020.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00024.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00028.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00032.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00036.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00040.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00044.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00048.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00052.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00056.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00060.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00064.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00068.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00072.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00076.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00080.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00084.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00088.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00092.jpg",
            "/data/vjuicefs_ai_camera_jgroup_research/public_data/11179416/data/VideoMatte240K_JPEG_HD/train/pha/0000/00096.jpg"
        ]
    },
}
'''