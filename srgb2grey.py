import cv2
import numpy as np

def convert_to_8bit_gray(input_path, output_path=None):
    """
    将 48-bit (16-bit/channel) 或 24-bit (8-bit/channel) 的 sRGB 图片转换为 8-bit 灰度图。
    
    参数:
        input_path (str): 输入图片的路径。
        output_path (str, optional): 输出图片的保存路径。
        
    返回:
        np.ndarray: 转换后的 8-bit 灰度图像素矩阵。
    """
    
    # 1. 读取图片 (保留原始深度和通道)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径是否正确: {input_path}")
        
    # 2. 转换为灰度图
    # 判断当前图片的通道数
    if len(img.shape) == 2:
        # 原图已经是单通道灰度图
        gray_img = img
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # 包含 Alpha 通道的 RGBA/BGRA 图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            # 标准 3 通道 RGB/BGR 图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}")

    # 3. 检查位深并统一转换为 8-bit
    if gray_img.dtype == np.uint16:
        # 对应 48-bit sRGB (16位 x 3通道)
        # 将 0-65535 的范围映射到 0-255
        gray_8bit = (gray_img / 256).astype(np.uint8)
    elif gray_img.dtype == np.uint8:
        # 对应 24-bit sRGB (8位 x 3通道)
        # 本身已经是 8-bit 取值范围 (0-255)，无需转换
        gray_8bit = gray_img
    else:
        # 防御性编程：处理其他奇怪的数据格式 (如 float32)
        print(f"警告: 遇到了非预期的位深格式 {gray_img.dtype}，强制归一化为 8-bit。")
        gray_8bit = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. 保存输出
    if output_path:
        cv2.imwrite(output_path, gray_8bit)
        # print(f"转换成功！图片已保存至: {output_path}") # 在流水线中为了控制台干净，可以注释掉这行
        
    return gray_8bit

# ================= 使用示例 =================
# input_file = "input_24bit_or_48bit.tif"
# output_file = "output_8bit_gray.tif" 
# gray_image = convert_to_8bit_gray(input_file, output_file)