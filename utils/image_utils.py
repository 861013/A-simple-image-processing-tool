#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像工具函数模块
提供图像处理相关的工具函数
"""

import cv2
import numpy as np
import os
from datetime import datetime


def validate_image_path(image_path):
    """
    验证图像文件路径
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        bool: 路径有效返回True，否则返回False
    """
    if not os.path.exists(image_path):
        return False
    
    # 检查文件扩展名
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    _, ext = os.path.splitext(image_path.lower())
    
    return ext in valid_extensions


def get_image_info(image):
    """
    获取图像信息
    
    Args:
        image (numpy.ndarray): 图像数组
        
    Returns:
        dict: 图像信息字典
    """
    if image is None:
        return None
    
    info = {
        'shape': image.shape,
        'dtype': image.dtype,
        'size': image.size,
        'channels': image.shape[2] if len(image.shape) == 3 else 1,
        'height': image.shape[0],
        'width': image.shape[1]
    }
    
    return info


def resize_image(image, target_size=None, scale_factor=None):
    """
    调整图像大小
    
    Args:
        image (numpy.ndarray): 输入图像
        target_size (tuple, optional): 目标尺寸 (width, height)
        scale_factor (float, optional): 缩放因子
        
    Returns:
        numpy.ndarray: 调整后的图像
    """
    if target_size is not None:
        return cv2.resize(image, target_size)
    elif scale_factor is not None:
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height))
    else:
        return image


def normalize_image(image, min_val=0, max_val=255):
    """
    归一化图像
    
    Args:
        image (numpy.ndarray): 输入图像
        min_val (int): 最小值
        max_val (int): 最大值
        
    Returns:
        numpy.ndarray: 归一化后的图像
    """
    # 转换为float32
    img_float = image.astype(np.float32)
    
    # 归一化到[0, 1]
    img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
    
    # 缩放到指定范围
    img_scaled = img_normalized * (max_val - min_val) + min_val
    
    return img_scaled.astype(np.uint8)


def convert_color_space(image, from_space, to_space):
    """
    转换颜色空间
    
    Args:
        image (numpy.ndarray): 输入图像
        from_space (str): 源颜色空间
        to_space (str): 目标颜色空间
        
    Returns:
        numpy.ndarray: 转换后的图像
    """
    color_conversions = {
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
        ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
        ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
        ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
        ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
        ('HSV', 'BGR'): cv2.COLOR_HSV2BGR
    }
    
    conversion_key = (from_space.upper(), to_space.upper())
    if conversion_key in color_conversions:
        return cv2.cvtColor(image, color_conversions[conversion_key])
    else:
        raise ValueError(f"不支持的颜色空间转换: {from_space} -> {to_space}")


def generate_filename(prefix="image", suffix="", extension=".jpg"):
    """
    生成文件名
    
    Args:
        prefix (str): 文件名前缀
        suffix (str): 文件名后缀
        extension (str): 文件扩展名
        
    Returns:
        str: 生成的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}{extension}"


def ensure_output_directory(output_path):
    """
    确保输出目录存在
    
    Args:
        output_path (str): 输出文件路径
        
    Returns:
        bool: 目录创建成功返回True，否则返回False
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建输出目录失败: {e}")
        return False


def calculate_image_similarity(img1, img2):
    """
    计算两张图像的相似度
    
    Args:
        img1 (numpy.ndarray): 第一张图像
        img2 (numpy.ndarray): 第二张图像
        
    Returns:
        float: 相似度分数 (0-1)
    """
    if img1.shape != img2.shape:
        return 0.0
    
    # 转换为灰度图
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = img1, img2
    
    # 计算结构相似性指数 (SSIM)
    from skimage.metrics import structural_similarity as ssim
    try:
        similarity = ssim(gray1, gray2)
        return max(0, similarity)  # 确保返回非负值
    except ImportError:
        # 如果skimage不可用，使用简单的MSE方法
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        max_pixel = 255.0
        similarity = 1 - (mse / (max_pixel ** 2))
        return max(0, similarity)


def apply_histogram_equalization(image):
    """
    应用直方图均衡化
    
    Args:
        image (numpy.ndarray): 输入图像
        
    Returns:
        numpy.ndarray: 均衡化后的图像
    """
    if len(image.shape) == 3:
        # 彩色图像
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        # 灰度图像
        return cv2.equalizeHist(image)


def create_noise_image(shape, noise_type='gaussian', **kwargs):
    """
    创建噪声图像
    
    Args:
        shape (tuple): 图像形状
        noise_type (str): 噪声类型 ('gaussian', 'salt_pepper', 'uniform')
        **kwargs: 噪声参数
        
    Returns:
        numpy.ndarray: 噪声图像
    """
    if noise_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 25)
        noise = np.random.normal(mean, std, shape)
    elif noise_type == 'salt_pepper':
        prob = kwargs.get('prob', 0.05)
        noise = np.random.random(shape)
        noise = np.where(noise < prob/2, -255, noise)
        noise = np.where(noise > 1 - prob/2, 255, noise)
        noise = np.where((noise >= prob/2) & (noise <= 1 - prob/2), 0, noise)
    elif noise_type == 'uniform':
        low = kwargs.get('low', -50)
        high = kwargs.get('high', 50)
        noise = np.random.uniform(low, high, shape)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    return noise.astype(np.uint8)
