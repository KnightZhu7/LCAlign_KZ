import cv2
import numpy as np

def convert_to_8bit_gray(input_path, output_path=None, channel_mode='all'):
    """
    将 48-bit 或 24-bit sRGB 图片转换为 8-bit 灰度图。
    支持全通道混合，或单独提取 R/G/B 通道。
    
    参数:
        input_path (str): 输入图片的路径。
        output_path (str, optional): 输出图片的保存路径。
        channel_mode (str): 转换模式。
                            'all' (默认) - 全通道加权转换为灰度图。
                            'R' - 仅提取红色通道。
                            'G' - 仅提取绿色通道。
                            'B' - 仅提取蓝色通道。
        
    返回:
        np.ndarray: 转换后的 8-bit 灰度图像素矩阵。
    """
    
    # 1. 读取图片 (保留原始深度和通道)
    # 注意：OpenCV 读取彩色图像的默认通道顺序是 B, G, R
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径是否正确: {input_path}")
        
    # 2. 转换为灰度图 或 提取指定通道
    channel_mode = channel_mode.upper()
    
    if len(img.shape) == 2:
        # 原图已经是单通道灰度图
        gray_img = img
        if channel_mode != 'ALL':
            print(f"警告: 输入已经是单通道灰度图，忽略参数 channel_mode='{channel_mode}'")
            
    elif len(img.shape) == 3:
        if channel_mode == 'ALL':
            # 全通道转换为灰度
            if img.shape[2] == 4:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channel_mode in ['B', 'G', 'R']:
            # 单独提取某个通道 (OpenCV 顺序为 BGR)
            channel_map = {'B': 0, 'G': 1, 'R': 2}
            ch_idx = channel_map[channel_mode]
            
            if ch_idx < img.shape[2]:
                gray_img = img[:, :, ch_idx]
            else:
                raise ValueError(f"图像通道数不足，无法提取 '{channel_mode}' 通道。")
        else:
            raise ValueError(f"不支持的 channel_mode: '{channel_mode}'。必须是 'all', 'R', 'G' 或 'B'。")
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}")

    # 3. 检查位深并统一转换为 8-bit
    if gray_img.dtype == np.uint16:
        # 对应 48-bit sRGB (16位) -> 将 0-65535 的范围映射到 0-255
        gray_8bit = (gray_img / 256).astype(np.uint8)
    elif gray_img.dtype == np.uint8:
        # 本身已经是 8-bit 取值范围，无需转换
        gray_8bit = gray_img
    else:
        # 防御性编程：处理其他数据格式
        print(f"警告: 遇到了非预期的位深格式 {gray_img.dtype}，强制归一化为 8-bit。")
        gray_8bit = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. 保存输出
    if output_path:
        cv2.imwrite(output_path, gray_8bit)
        
    return gray_8bit

# ================= 使用示例 =================
# input_file = "input.tif"
# output_file_all = "output_all.tif"
# output_file_R = "output_R.tif"

# 1. 默认全通道混合灰度
# img_gray = convert_to_8bit_gray(input_file, output_file_all, channel_mode='all')

# 2. 仅提取红色通道作为灰度图
# img_R_gray = convert_to_8bit_gray(input_file, output_file_R, channel_mode='R')
