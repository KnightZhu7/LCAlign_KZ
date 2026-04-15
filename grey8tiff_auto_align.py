import cv2
import numpy as np
import os
from typing import List, Tuple

# ─────────────────────────────────────────────
# 1. 读取图像
# ─────────────────────────────────────────────
def load_images(paths: List[str]) -> List[np.ndarray]:
    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取: {p}")
            
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if img.dtype != np.uint8:
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
                
        images.append(img)
    return images

# ─────────────────────────────────────────────
# 2. 交互式选点工具 (GUI)
# ─────────────────────────────────────────────
def get_manual_points(img_gray: np.ndarray, window_name: str, num_points: int = 3) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_vis = clahe.apply(img_gray)
    img_display = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    img_clean = img_display.copy()
    
    pts = []

    def click_event(event, x, y, flags, param):
        nonlocal pts, img_display
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < num_points:
                pts.append([x, y])
                cv2.circle(img_display, (x, y), 20, (0, 0, 255), -1)
                cv2.putText(img_display, str(len(pts)), (x+30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                cv2.imshow(window_name, img_display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700) 
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, img_display)
    
    print(f"\n[{window_name}] 请点击 {num_points} 个特征点 (按 'R' 键重置)。")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if len(pts) == num_points:
            cv2.waitKey(600) 
            break
        if key == ord('r') or key == ord('R'):
            pts = []
            img_display = img_clean.copy()
            cv2.imshow(window_name, img_display)
            print("  -> 已重置点位，请重新点击。")

    cv2.destroyWindow(window_name)
    cv2.waitKey(1) 
    
    return np.float32(pts)

# ─────────────────────────────────────────────
# 3. 核心算法：纯刚性变换 (仅平移 + 旋转)
# ─────────────────────────────────────────────
def get_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    使用 Kabsch 算法计算绝对刚性变换矩阵。
    严格保证图像尺寸不发生任何缩放或拉伸，仅有平移(tx, ty)和旋转(theta)。
    """
    # 1. 计算质心
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    # 2. 将点集去中心化 (移到原点)
    src_centered = src_pts - centroid_src
    dst_centered = dst_pts - centroid_dst

    # 3. 计算协方差矩阵
    H = np.dot(src_centered.T, dst_centered)

    # 4. 奇异值分解 (SVD) 求最优旋转矩阵
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 消除镜像反射的可能 (确保是正常的旋转)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 5. 计算平移向量
    t = centroid_dst.T - np.dot(R, centroid_src.T)

    # 6. 组合成 2x3 的标准 OpenCV 仿射矩阵
    M = np.empty((2, 3), dtype=np.float32)
    M[:, :2] = R
    M[:, 2] = t
    return M

# ─────────────────────────────────────────────
# 4. 基于手动选点和纯刚性算法的对齐
# ─────────────────────────────────────────────
def align_all(images: List[np.ndarray], ref_index: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ref = images[ref_index]
    h, w = ref.shape[:2]
    
    aligned_images = []
    aligned_masks = []
    base_mask = np.ones((h, w), dtype=np.uint8) * 255

    print("="*50)
    print(" 第一步：设置参考基准点 (取 3 个点)")
    print("="*50)
    ref_pts = get_manual_points(ref, "Step 1: Reference Image", num_points=3)
    
    # 制作小抄
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ref_guide = cv2.cvtColor(clahe.apply(ref), cv2.COLOR_GRAY2BGR)
    for i, pt in enumerate(ref_pts):
        cv2.circle(ref_guide, (int(pt[0]), int(pt[1])), 20, (0, 255, 0), -1)
        cv2.putText(ref_guide, str(i+1), (int(pt[0])+30, int(pt[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
    cv2.namedWindow("Reference Guide (DO NOT CLOSE)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reference Guide (DO NOT CLOSE)", 600, 450)
    cv2.imshow("Reference Guide (DO NOT CLOSE)", ref_guide)
    cv2.waitKey(1)

    print("\n" + "="*50)
    print(" 第二步：开始逐张配准目标图 (严格保持原比例)")
    print("="*50)
    
    for i, img in enumerate(images):
        if i == ref_index:
            aligned_images.append(ref)
            aligned_masks.append(base_mask)
            print(f"  > 第 {i} 张图片 (参考图) 已跳过。")
            continue
        
        tgt_pts = get_manual_points(img, f"Step 2: Target Image {i}/{len(images)-1}", num_points=3)
        
        # 【核心修改点】调用纯刚性变换算法，取代了 getAffineTransform
        M = get_rigid_transform(tgt_pts, ref_pts)
        
        H = np.eye(3, dtype=np.float32)
        H[0:2, :] = M
        
        warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = cv2.warpPerspective(base_mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        aligned_images.append(warped_img)
        aligned_masks.append(warped_mask)
        print(f"  > 第 {i} 张图片刚性配准完成！")

    cv2.destroyWindow("Reference Guide (DO NOT CLOSE)")
    cv2.waitKey(1)
    
    return aligned_images, aligned_masks

# ─────────────────────────────────────────────
# 5. 计算有效掩码与裁剪
# ─────────────────────────────────────────────
def compute_overlap_mask(masks: List[np.ndarray]) -> np.ndarray:
    final_mask = np.ones(masks[0].shape[:2], dtype=np.uint8) * 255
    for m in masks:
        final_mask = cv2.bitwise_and(final_mask, m)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(final_mask, kernel, iterations=2)

def largest_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    bin_mask = (mask == 255).astype(np.int32)
    rows, cols = bin_mask.shape
    heights = np.zeros(cols, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, 0, 0)

    for r in range(rows):
        heights = np.where(bin_mask[r] == 1, heights + 1, 0)
        stack = [] 
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, ht = stack.pop()
                w = c - idx
                area = ht * w
                if area > best_area:
                    best_area = area
                    best_rect = (idx, r - ht + 1, w, ht)
                start = idx
            stack.append((start, h))

    return best_rect

def process(image_paths: List[str], output_dir: str = ".", ref_index: int = 0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = load_images(image_paths)
    print(f"\n  > 成功加载 {len(images)} 张图像 | 尺寸: {images[0].shape} | 类型: {images[0].dtype}")

    aligned, masks = align_all(images, ref_index)
    
    overlap_mask = compute_overlap_mask(masks)
    x, y, w, h = largest_inscribed_rectangle(overlap_mask)
    
    if w <= 0 or h <= 0:
        x, y, w, h = 0, 0, images[0].shape[1], images[0].shape[0]

    results = []
    for i, img in enumerate(aligned):
        crop = img[y:y+h, x:x+w]
        out_path = os.path.join(output_dir, f"aligned_crop_{i:02d}.tif")
        cv2.imwrite(out_path, crop)
        results.append(crop)
    
    print(f"  > 对齐裁剪后的图片已保存至: {output_dir}\n")
    return results, (x, y, w, h)