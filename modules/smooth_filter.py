#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平滑滤波模块
实现各种平滑滤波算法
"""

import cv2
import numpy as np
from .base_processor import BaseImageProcessor


class SmoothFilter(BaseImageProcessor):
    """平滑滤波处理类"""
    
    def __init__(self):
        super().__init__()
    
    def gaussian_blur(self, image, kernel_size=5, sigma=0):
        """
        高斯模糊
        
        Args:
            image (numpy.ndarray): 输入图像
            kernel_size (int): 核大小，必须是奇数
            sigma (float): 高斯标准差，0表示自动计算
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保核大小为奇数
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def mean_filter(self, image, kernel_size=5):
        """
        均值滤波
        
        Args:
            image (numpy.ndarray): 输入图像
            kernel_size (int): 核大小
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)
    
    def median_filter(self, image, kernel_size=5):
        """
        中值滤波
        
        Args:
            image (numpy.ndarray): 输入图像
            kernel_size (int): 核大小
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)
    
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        双边滤波（保边去噪）
        
        Args:
            image (numpy.ndarray): 输入图像
            d (int): 像素邻域直径
            sigma_color (float): 颜色空间的标准差
            sigma_space (float): 坐标空间的标准差
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def process(self, method='gaussian', **kwargs):
        """
        执行平滑滤波处理
        
        Args:
            method (str): 滤波方法 ('gaussian', 'mean', 'median', 'bilateral')
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'gaussian':
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 0)
            self.processed_image = self.gaussian_blur(image, kernel_size, sigma)
        elif method == 'mean':
            kernel_size = kwargs.get('kernel_size', 5)
            self.processed_image = self.mean_filter(image, kernel_size)
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            self.processed_image = self.median_filter(image, kernel_size)
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            self.processed_image = self.bilateral_filter(image, d, sigma_color, sigma_space)
        else:
            raise ValueError(f"不支持的平滑滤波方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的平滑滤波方法"""
        return ['gaussian', 'mean', 'median', 'bilateral']
