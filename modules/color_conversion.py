#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色空间转换模块
支持RGB、RGBA、HSV、LAB、YUV、CMYK、灰度等多种颜色空间转换
"""

import cv2
import numpy as np
from PIL import Image
from .base_processor import BaseImageProcessor


class ColorConversion(BaseImageProcessor):
    """颜色空间转换处理类"""
    
    def __init__(self):
        super().__init__()
        self.supported_color_spaces = {
            'RGB': 'RGB',
            'RGBA': 'RGBA', 
            'HSV': 'HSV',
            'LAB': 'LAB',
            'YUV': 'YUV',
            'CMYK': 'CMYK',
            'GRAY': 'GRAY',
            'L': 'L',  # PIL灰度模式
            'P': 'P',  # PIL调色板模式
            '1': '1',  # PIL黑白模式
            'LA': 'LA',  # PIL灰度+Alpha
            'RGBX': 'RGBX',  # PIL RGB+填充
            'RGBA': 'RGBA',  # PIL RGBA
            'RGBa': 'RGBa',  # PIL RGB+预乘Alpha
            'LUV': 'LUV',  # OpenCV LUV
            'XYZ': 'XYZ',  # OpenCV XYZ
            'YCrCb': 'YCrCb'  # OpenCV YCrCb
        }
    
    def detect_color_space(self, image):
        """
        检测图像的颜色空间
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            str: 检测到的颜色空间
        """
        if image is None:
            return "未知"
        
        if len(image.shape) == 2:
            return "GRAY"
        elif len(image.shape) == 3:
            channels = image.shape[2]
            if channels == 1:
                return "GRAY"
            elif channels == 3:
                return "RGB"
            elif channels == 4:
                return "RGBA"
            else:
                return f"多通道({channels})"
        else:
            return "未知"
    
    def convert_to_rgb(self, image, from_space='RGB'):
        """
        将图像转换为RGB格式
        
        Args:
            image (numpy.ndarray): 输入图像
            from_space (str): 源颜色空间
            
        Returns:
            numpy.ndarray: RGB格式的图像
        """
        if image is None:
            return None
        
        if from_space.upper() == 'RGB':
            return image.copy()
        elif from_space.upper() == 'RGBA':
            # RGBA转RGB，使用白色背景
            if len(image.shape) == 3 and image.shape[2] == 4:
                rgb = image[:, :, :3]
                alpha = image[:, :, 3:4] / 255.0
                background = np.ones_like(rgb) * 255
                rgb_result = rgb * alpha + background * (1 - alpha)
                return rgb_result.astype(np.uint8)
            return image[:, :, :3]
        elif from_space.upper() == 'GRAY':
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif from_space.upper() == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif from_space.upper() == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        elif from_space.upper() == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        elif from_space.upper() == 'YCRCB':
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        elif from_space.upper() == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
        elif from_space.upper() == 'XYZ':
            return cv2.cvtColor(image, cv2.COLOR_XYZ2RGB)
        else:
            print(f"不支持的颜色空间转换: {from_space} -> RGB")
            return image.copy()
    
    def convert_to_grayscale(self, image, method='luminance'):
        """
        将图像转换为灰度图
        
        Args:
            image (numpy.ndarray): 输入图像
            method (str): 转换方法 ('luminance', 'average', 'max', 'min')
            
        Returns:
            numpy.ndarray: 灰度图像
        """
        if image is None:
            return None
        
        if len(image.shape) == 2:
            return image.copy()
        
        if len(image.shape) == 3:
            if method == 'luminance':
                # 使用亮度公式: 0.299*R + 0.587*G + 0.114*B
                if image.shape[2] == 3:
                    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
                else:  # RGBA
                    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            elif method == 'average':
                gray = np.mean(image[..., :3], axis=2)
            elif method == 'max':
                gray = np.max(image[..., :3], axis=2)
            elif method == 'min':
                gray = np.min(image[..., :3], axis=2)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            return gray.astype(np.uint8)
        
        return image
    
    def convert_cmyk_to_rgb(self, cmyk_image):
        """
        将CMYK图像转换为RGB
        
        Args:
            cmyk_image (numpy.ndarray): CMYK图像
            
        Returns:
            numpy.ndarray: RGB图像
        """
        if cmyk_image is None or len(cmyk_image.shape) != 3 or cmyk_image.shape[2] != 4:
            return None
        
        # CMYK值范围通常是0-100，需要归一化到0-1
        cmyk = cmyk_image.astype(np.float32) / 100.0
        
        # CMYK到RGB转换公式
        rgb = np.zeros_like(cmyk)
        rgb[:, :, 0] = 255 * (1 - cmyk[:, :, 0]) * (1 - cmyk[:, :, 3])  # R
        rgb[:, :, 1] = 255 * (1 - cmyk[:, :, 1]) * (1 - cmyk[:, :, 3])  # G
        rgb[:, :, 2] = 255 * (1 - cmyk[:, :, 2]) * (1 - cmyk[:, :, 3])  # B
        
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def convert_rgb_to_cmyk(self, rgb_image):
        """
        将RGB图像转换为CMYK
        
        Args:
            rgb_image (numpy.ndarray): RGB图像
            
        Returns:
            numpy.ndarray: CMYK图像
        """
        if rgb_image is None or len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            return None
        
        rgb = rgb_image.astype(np.float32) / 255.0
        
        # RGB到CMYK转换
        cmyk = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.float32)
        
        # 计算K (黑色)
        k = 1 - np.max(rgb, axis=2)
        
        # 避免除零错误
        k_safe = np.where(k == 1, 0.0001, k)
        
        # 计算CMY
        cmyk[:, :, 0] = (1 - rgb[:, :, 0] - k) / (1 - k_safe)  # C
        cmyk[:, :, 1] = (1 - rgb[:, :, 1] - k) / (1 - k_safe)  # M
        cmyk[:, :, 2] = (1 - rgb[:, :, 2] - k) / (1 - k_safe)  # Y
        cmyk[:, :, 3] = k  # K
        
        # 转换为0-100范围
        cmyk = np.clip(cmyk, 0, 1) * 100
        
        return cmyk.astype(np.uint8)
    
    def enhance_color_image(self, image, enhancement_factor=1.2):
        """
        增强彩色图像
        
        Args:
            image (numpy.ndarray): 输入图像
            enhancement_factor (float): 增强因子
            
        Returns:
            numpy.ndarray: 增强后的图像
        """
        if image is None:
            return None
        
        # 转换到HSV空间进行增强
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 增强饱和度和亮度
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * enhancement_factor, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * enhancement_factor, 0, 255)
            
            # 转换回RGB
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return enhanced
        else:
            # 灰度图像增强
            enhanced = np.clip(image * enhancement_factor, 0, 255)
            return enhanced.astype(np.uint8)
    
    def process(self, method='convert_to_rgb', **kwargs):
        """
        执行颜色空间转换处理
        
        Args:
            method (str): 处理方法
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'convert_to_rgb':
            from_space = kwargs.get('from_space', 'RGB')
            self.processed_image = self.convert_to_rgb(image, from_space)
        elif method == 'convert_to_grayscale':
            gray_method = kwargs.get('gray_method', 'luminance')
            self.processed_image = self.convert_to_grayscale(image, gray_method)
        elif method == 'convert_cmyk_to_rgb':
            self.processed_image = self.convert_cmyk_to_rgb(image)
        elif method == 'convert_rgb_to_cmyk':
            self.processed_image = self.convert_rgb_to_cmyk(image)
        elif method == 'enhance_color':
            enhancement_factor = kwargs.get('enhancement_factor', 1.2)
            self.processed_image = self.enhance_color_image(image, enhancement_factor)
        elif method == 'convert_color_space':
            target_space = kwargs.get('color_space', 'HSV')
            if target_space.upper() == 'HSV':
                self.processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif target_space.upper() == 'LAB':
                self.processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif target_space.upper() == 'YUV':
                self.processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif target_space.upper() == 'GRAY':
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                self.processed_image = image.copy()
        else:
            raise ValueError(f"不支持的颜色转换方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的颜色转换方法"""
        return [
            'convert_to_rgb', 'convert_to_grayscale', 'convert_cmyk_to_rgb',
            'convert_rgb_to_cmyk', 'enhance_color', 'convert_color_space'
        ]
    
    def get_supported_color_spaces(self):
        """获取支持的颜色空间列表"""
        return list(self.supported_color_spaces.keys())