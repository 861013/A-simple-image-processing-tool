#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锐化滤波模块
实现各种锐化滤波算法
"""

import cv2
import numpy as np
from .base_processor import BaseImageProcessor


class SharpenFilter(BaseImageProcessor):
    """锐化滤波处理类"""
    
    def __init__(self):
        super().__init__()
    
    def unsharp_mask(self, image, strength=1.0, radius=1.0, threshold=0):
        """
        反锐化掩模
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 锐化强度
            radius (float): 模糊半径
            threshold (int): 阈值
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 创建高斯模糊版本
        kernel_size = max(3, int(radius * 2) * 2 + 1)  # 确保核大小为奇数
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)
        
        # 计算锐化掩模
        mask = image.astype(np.float32) - blurred.astype(np.float32)
        
        # 应用阈值
        if threshold > 0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
        # 应用锐化
        sharpened = image.astype(np.float32) + strength * mask
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def laplacian_sharpen(self, image, strength=1.0):
        """
        拉普拉斯锐化
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 锐化强度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 拉普拉斯核
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]) * strength
        
        # 应用锐化核
        sharpened = cv2.filter2D(image, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def high_pass_sharpen(self, image, strength=1.0):
        """
        高通锐化
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 锐化强度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 高通核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        # 应用锐化核
        sharpened = cv2.filter2D(image, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def sobel_sharpen(self, image, strength=1.0):
        """
        Sobel锐化
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 锐化强度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 计算Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 归一化梯度
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        
        # 应用锐化
        if len(image.shape) == 3:
            gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2RGB)
        
        sharpened = image.astype(np.float32) + strength * gradient_magnitude.astype(np.float32)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def process(self, method='unsharp_mask', **kwargs):
        """
        执行锐化滤波处理
        
        Args:
            method (str): 锐化方法 ('unsharp_mask', 'laplacian', 'high_pass', 'sobel')
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'unsharp_mask':
            strength = kwargs.get('strength', 1.0)
            radius = kwargs.get('radius', 1.0)
            threshold = kwargs.get('threshold', 0)
            self.processed_image = self.unsharp_mask(image, strength, radius, threshold)
        elif method == 'laplacian':
            strength = kwargs.get('strength', 1.0)
            self.processed_image = self.laplacian_sharpen(image, strength)
        elif method == 'high_pass':
            strength = kwargs.get('strength', 1.0)
            self.processed_image = self.high_pass_sharpen(image, strength)
        elif method == 'sobel':
            strength = kwargs.get('strength', 1.0)
            self.processed_image = self.sobel_sharpen(image, strength)
        else:
            raise ValueError(f"不支持的锐化方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的锐化方法"""
        return ['unsharp_mask', 'laplacian', 'high_pass', 'sobel']
