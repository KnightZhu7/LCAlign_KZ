import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import re  # 【新增】引入正则表达式模块，用于自然排序

# 导入已有的功能模块
from srgb2grey import convert_to_8bit_gray
from grey8tiff_auto_align import process as align_process, load_images

def natural_sort_key(s):
    """【新增】将字符串中的数字提取出来转化为整数进行真实大小比对 (自然排序)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_image_paths(directory):
    """辅助函数：获取指定目录下所有 TIFF 图片路径并按自然顺序排序"""
    if not os.path.exists(directory):
        return []
    exts = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
        
    # 【核心修复】：使用 natural_sort_key 替代普通的字母表排序
    return sorted(paths, key=natural_sort_key)

def main():
    print("="*60)
    print(" 图像转换、对齐与 ROI 灰度动态变化分析管道 ")
    print("="*60)
    
    # 1. 获取并检查目录
    script_root_dir = os.path.dirname(os.path.abspath(__file__))
    orgimg_dir = os.path.join(script_root_dir, "orgimg")
    interim_dir = os.path.join(script_root_dir, "8bit_gray") 
    aligned_dir = os.path.join(script_root_dir, "relimg")   
    hist_dir = os.path.join(script_root_dir, "histograms")  
    
    # 最终汇总数据和折线图保存在根目录，方便查找
    csv_output_path = os.path.join(script_root_dir, "roi_averages_trend.csv") 
    trend_plot_path = os.path.join(script_root_dir, "roi_averages_trend_chart.png")
    
    org_paths = get_image_paths(orgimg_dir)
    if not org_paths:
        print(f"错误：在 '{orgimg_dir}' 文件夹下没有找到原 TIFF 文件。")
        return
        
    print(f"\n[检测] 找到 {len(org_paths)} 张原始图片。")
    
    # 2. 检查 relimg 是否已有对齐图片
    rel_paths = get_image_paths(aligned_dir)
    skip_alignment = False
    
    if len(rel_paths) > 0 and len(rel_paths) == len(org_paths):
        print(f"[检测] 发现 '{aligned_dir}' 文件夹中已存在 {len(rel_paths)} 张对齐好的图片！")
        ans = input("  -> 是否跳过转换和配准，直接使用现有对齐图片进行分析？(Y/N) [默认 Y]: ").strip().upper()
        if ans != 'N':
            skip_alignment = True

    aligned_images = []
    ref_idx = 0
    
    # 3. 分支处理：跳过 or 重新配准
    if skip_alignment:
        print("\n[加载] 正在读取现有的对齐图片...")
        aligned_images = load_images(rel_paths)
        ref_idx_str = input(f"请选择一张图片用于框选 ROI (0-{len(aligned_images)-1}) [默认 0]: ").strip()
        ref_idx = int(ref_idx_str) if ref_idx_str.isdigit() else 0
        ref_idx = max(0, min(ref_idx, len(aligned_images)-1))
    else:
        ref_idx_str = input(f"\n请选择要作为【对齐参照】的原图序号 (0-{len(org_paths)-1}) [默认 0]: ").strip()
        ref_idx = int(ref_idx_str) if ref_idx_str.isdigit() else 0
        ref_idx = max(0, min(ref_idx, len(org_paths)-1))
        
        ch_mode = input("\n请选择转换模式 (all: 全通道混合, R: 仅红通道, G: 仅绿通道, B: 仅蓝通道) [默认 all]: ").strip().upper()
        
        if ch_mode == 'R':
            mode_desc = "仅提取 🔴 红 (Red) 通道"
        elif ch_mode == 'G':
            mode_desc = "仅提取 🟢 绿 (Green) 通道"
        elif ch_mode == 'B':
            mode_desc = "仅提取 🔵 蓝 (Blue) 通道"
        else:
            ch_mode = 'ALL'
            mode_desc = "全通道混合 (心理视觉加权)"

        print(f"  -> 已确认！当前使用模式：【 {mode_desc} 】")

        os.makedirs(interim_dir, exist_ok=True)
        print(f"\n[处理] 开始进行 8-bit 灰度图转换...")
        converted_paths = []
        for p in org_paths:
            out_path = os.path.join(interim_dir, "8bit_" + os.path.basename(p))
            convert_to_8bit_gray(p, out_path) 
            converted_paths.append(out_path)
            
        print("\n[处理] 开始图像配准...")
        aligned_images, _ = align_process(converted_paths, output_dir=aligned_dir, ref_index=ref_idx)

    # 4. 框选 ROI
    print("="*60)
    print(" 正在打开参照图以供框选 ROI...")
    print(" 【操作说明】画完所有需要的框（如2个）后，按 ESC 结束")
    print("="*60)
    
    ref_img = aligned_images[ref_idx]
    window_name = "Select ROIs - SPACE to confirm each, ESC to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    rois = cv2.selectROIs(window_name, ref_img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  
    
    if len(rois) == 0:
        print("\n[警告] 未选择任何有效区域。")
        return
    
    # 5. 计算、绘图并采集数据
    print("\n[分析] 开始提取数据并生成直方图...")
    os.makedirs(hist_dir, exist_ok=True)
    
    csv_data = []
    trend_data_series = {roi_idx: [] for roi_idx in range(len(rois))}
    colors = ['dodgerblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple', 'gold']
    
    for i, img in enumerate(aligned_images):
        orig_name = os.path.basename(org_paths[i])
        
        for roi_idx, roi in enumerate(rois):
            x, y, w, h = roi
            crop = img[y:y+h, x:x+w]
            
            hist = cv2.calcHist([crop], [0], None, [256], [0, 256]).flatten()
            avg_gray = np.mean(crop)
            
            # 记录数据用于 CSV 和趋势折线图
            csv_data.append([i, orig_name, roi_idx + 1, f"{avg_gray:.4f}"])
            trend_data_series[roi_idx].append(avg_gray)
            
            # 绘制独立直方图
            plt.figure(figsize=(10, 5))
            c = colors[roi_idx % len(colors)] 
            plt.bar(range(256), hist, color=c, alpha=0.85, width=1.0)
            
            title_suffix = " (Ref)" if i == ref_idx else ""
            plt.title(f"Histogram - [{i:02d}] {orig_name} - ROI {roi_idx + 1}{title_suffix}")
            plt.xlabel("Pixel Value (0-255)")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.grid(axis='y', linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            save_name = f"hist_img_{i:02d}_roi_{roi_idx+1}.png"
            plt.savefig(os.path.join(hist_dir, save_name), dpi=150) 
            plt.close()
            
        print(f"  > 已处理: {orig_name}")

    # 6. 写入 CSV 文件
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_Index', 'Image_Name', 'ROI_Index', 'Average_Grayscale'])
        writer.writerows(csv_data)
        
    # 7. 绘制平均灰度变化趋势折线图
    print("\n[绘图] 正在生成灰度趋势演变折线图...")
    plt.figure(figsize=(12, 6))
    
    x_axis = range(len(aligned_images))
    for roi_idx, avg_values in trend_data_series.items():
        c = colors[roi_idx % len(colors)]
        plt.plot(x_axis, avg_values, marker='o', markersize=5, linewidth=2.5, 
                 color=c, label=f'ROI {roi_idx + 1}')

    plt.title("Average Grayscale Trend Across Image Sequence", fontsize=14, pad=15)
    plt.xlabel("Image Index (Time / Sequence)", fontsize=12)
    plt.ylabel("Average Grayscale Value (0-255)", fontsize=12)
    
    plt.xticks(x_axis, [str(i) for i in x_axis], rotation=45 if len(x_axis) > 20 else 0)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(trend_plot_path, dpi=200) 
    plt.close()
    
    print(f"\n============================================================")
    print(f" 全部处理完成！你的成果如下：")
    print(f" 1. 独立直方图文件夹: {hist_dir}")
    print(f" 2. 定量数据表格: {csv_output_path}")
    print(f" 3. 灰度变化趋势图: {trend_plot_path}")
    print(f"============================================================")

if __name__ == "__main__":
    main()