#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
色散效果模块
实现各种色散效果算法
"""

import cv2
import numpy as np
from .base_processor import BaseImageProcessor


class ChromaticAberration(BaseImageProcessor):
    """色散效果处理类"""
    
    def __init__(self):
        super().__init__()
    
    def basic_chromatic_aberration(self, image, intensity=5, direction='horizontal'):
        """
        基础色散效果
        
        Args:
            image (numpy.ndarray): 输入图像
            intensity (int): 色散强度
            direction (str): 色散方向 ('horizontal', 'vertical', 'radial')
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if len(image.shape) != 3:
            return image
        
        # 分离RGB通道
        r, g, b = cv2.split(image)
        rows, cols = r.shape[:2]
        
        if direction == 'horizontal':
            # 水平色散
            M_r = np.float32([[1, 0, intensity], [0, 1, 0]])
            M_g = np.float32([[1, 0, 0], [0, 1, 0]])
            M_b = np.float32([[1, 0, -intensity], [0, 1, 0]])
        elif direction == 'vertical':
            # 垂直色散
            M_r = np.float32([[1, 0, 0], [0, 1, intensity]])
            M_g = np.float32([[1, 0, 0], [0, 1, 0]])
            M_b = np.float32([[1, 0, 0], [0, 1, -intensity]])
        else:  # radial
            # 径向色散
            center_x, center_y = cols // 2, rows // 2
            M_r = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
            M_r[0, 2] += intensity
            M_g = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
            M_b = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
            M_b[0, 2] -= intensity
        
        # 应用变换
        r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
        g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows))
        
        # 合并通道
        result = cv2.merge([r_shifted, g_shifted, b_shifted])
        
        return result
    
    def advanced_chromatic_aberration(self, image, intensity=5, blur_radius=1):
        """
        高级色散效果（带模糊）
        
        Args:
            image (numpy.ndarray): 输入图像
            intensity (int): 色散强度
            blur_radius (int): 模糊半径
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if len(image.shape) != 3:
            return image
        
        # 分离RGB通道
        r, g, b = cv2.split(image)
        rows, cols = r.shape[:2]
        
        # 创建变换矩阵
        M_r = np.float32([[1, 0, intensity], [0, 1, 0]])
        M_g = np.float32([[1, 0, 0], [0, 1, 0]])
        M_b = np.float32([[1, 0, -intensity], [0, 1, 0]])
        
        # 应用变换
        r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
        g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows))
        
        # 应用模糊效果
        if blur_radius > 0:
            kernel_size = blur_radius * 2 + 1
            r_shifted = cv2.GaussianBlur(r_shifted, (kernel_size, kernel_size), 0)
            b_shifted = cv2.GaussianBlur(b_shifted, (kernel_size, kernel_size), 0)
        
        # 合并通道
        result = cv2.merge([r_shifted, g_shifted, b_shifted])
        
        return result
    
    def radial_chromatic_aberration(self, image, intensity=5, center=None):
        """
        径向色散效果
        
        Args:
            image (numpy.ndarray): 输入图像
            intensity (int): 色散强度
            center (tuple): 色散中心点，默认为图像中心
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if len(image.shape) != 3:
            return image
        
        rows, cols = image.shape[:2]
        if center is None:
            center = (cols // 2, rows // 2)
        
        # 分离RGB通道
        r, g, b = cv2.split(image)
        
        # 创建径向色散效果
        y, x = np.ogrid[:rows, :cols]
        cx, cy = center
        
        # 计算到中心的距离
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_distance = np.sqrt(cx**2 + cy**2)
        
        # 归一化距离
        normalized_distance = distance / max_distance
        
        # 创建色散偏移
        r_offset = intensity * normalized_distance
        b_offset = -intensity * normalized_distance
        
        # 应用偏移
        r_shifted = np.zeros_like(r)
        b_shifted = np.zeros_like(b)
        
        for i in range(rows):
            for j in range(cols):
                r_offset_x = int(j + r_offset[i, j])
                b_offset_x = int(j + b_offset[i, j])
                
                if 0 <= r_offset_x < cols:
                    r_shifted[i, j] = r[i, r_offset_x]
                else:
                    r_shifted[i, j] = r[i, j]
                
                if 0 <= b_offset_x < cols:
                    b_shifted[i, j] = b[i, b_offset_x]
                else:
                    b_shifted[i, j] = b[i, j]
        
        # 合并通道
        result = cv2.merge([r_shifted, g, b_shifted])
        
        return result
    
    def prism_effect(self, image, intensity=5, angle=0):
        """
        棱镜效果
        
        Args:
            image (numpy.ndarray): 输入图像
            intensity (int): 效果强度
            angle (float): 棱镜角度（度）
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if len(image.shape) != 3:
            return image
        
        # 分离RGB通道
        r, g, b = cv2.split(image)
        rows, cols = r.shape[:2]
        
        # 将角度转换为弧度
        angle_rad = np.radians(angle)
        
        # 创建变换矩阵
        M_r = np.float32([[1, 0, intensity * np.cos(angle_rad)], 
                         [0, 1, intensity * np.sin(angle_rad)]])
        M_g = np.float32([[1, 0, 0], [0, 1, 0]])
        M_b = np.float32([[1, 0, -intensity * np.cos(angle_rad)], 
                         [0, 1, -intensity * np.sin(angle_rad)]])
        
        # 应用变换
        r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
        g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows))
        
        # 合并通道
        result = cv2.merge([r_shifted, g_shifted, b_shifted])
        
        return result
    
    def process(self, method='basic', **kwargs):
        """
        执行色散效果处理
        
        Args:
            method (str): 色散方法 ('basic', 'advanced', 'radial', 'prism')
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'basic':
            intensity = kwargs.get('intensity', 5)
            direction = kwargs.get('direction', 'horizontal')
            self.processed_image = self.basic_chromatic_aberration(image, intensity, direction)
        elif method == 'advanced':
            intensity = kwargs.get('intensity', 5)
            blur_radius = kwargs.get('blur_radius', 1)
            self.processed_image = self.advanced_chromatic_aberration(image, intensity, blur_radius)
        elif method == 'radial':
            intensity = kwargs.get('intensity', 5)
            center = kwargs.get('center', None)
            self.processed_image = self.radial_chromatic_aberration(image, intensity, center)
        elif method == 'prism':
            intensity = kwargs.get('intensity', 5)
            angle = kwargs.get('angle', 0)
            self.processed_image = self.prism_effect(image, intensity, angle)
        else:
            raise ValueError(f"不支持的色散方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的色散方法"""
        return ['basic', 'advanced', 'radial', 'prism']
