#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘检测模块
实现各种边缘检测算法
"""

import cv2
import numpy as np
from .base_processor import BaseImageProcessor


class EdgeDetection(BaseImageProcessor):
    """边缘检测处理类"""
    
    def __init__(self):
        super().__init__()
    
    def canny_edge(self, image, low_threshold=50, high_threshold=150, aperture_size=3):
        """
        Canny边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            low_threshold (int): 低阈值
            high_threshold (int): 高阈值
            aperture_size (int): Sobel核大小
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Canny边缘检测
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
        
        # 转换回RGB
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def sobel_edge(self, image, dx=1, dy=1, ksize=3):
        """
        Sobel边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            dx (int): x方向导数阶数
            dy (int): y方向导数阶数
            ksize (int): Sobel核大小
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Sobel边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize)
        
        # 计算梯度幅值
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # 转换回RGB
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
    def laplacian_edge(self, image, ksize=3):
        """
        Laplacian边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            ksize (int): Laplacian核大小
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Laplacian边缘检测
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 转换回RGB
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    
    def prewitt_edge(self, image):
        """
        Prewitt边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Prewitt核
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # 应用Prewitt核
        edges_x = cv2.filter2D(gray, -1, prewitt_x)
        edges_y = cv2.filter2D(gray, -1, prewitt_y)
        
        # 计算梯度幅值
        magnitude = np.sqrt(edges_x**2 + edges_y**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # 转换回RGB
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
    def roberts_edge(self, image):
        """
        Roberts边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Roberts核
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        
        # 应用Roberts核
        edges_x = cv2.filter2D(gray, -1, roberts_x)
        edges_y = cv2.filter2D(gray, -1, roberts_y)
        
        # 计算梯度幅值
        magnitude = np.sqrt(edges_x**2 + edges_y**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # 转换回RGB
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
    def scharr_edge(self, image, dx=1, dy=1):
        """
        Scharr边缘检测
        
        Args:
            image (numpy.ndarray): 输入图像
            dx (int): x方向导数阶数
            dy (int): y方向导数阶数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Scharr边缘检测
        scharrx = cv2.Scharr(gray, cv2.CV_64F, dx, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, dy)
        
        # 计算梯度幅值
        magnitude = np.sqrt(scharrx**2 + scharry**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # 转换回RGB
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
    def process(self, method='canny', **kwargs):
        """
        执行边缘检测处理
        
        Args:
            method (str): 边缘检测方法
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            aperture_size = kwargs.get('aperture_size', 3)
            self.processed_image = self.canny_edge(image, low_threshold, high_threshold, aperture_size)
        elif method == 'sobel':
            dx = kwargs.get('dx', 1)
            dy = kwargs.get('dy', 1)
            ksize = kwargs.get('ksize', 3)
            self.processed_image = self.sobel_edge(image, dx, dy, ksize)
        elif method == 'laplacian':
            ksize = kwargs.get('ksize', 3)
            self.processed_image = self.laplacian_edge(image, ksize)
        elif method == 'prewitt':
            self.processed_image = self.prewitt_edge(image)
        elif method == 'roberts':
            self.processed_image = self.roberts_edge(image)
        elif method == 'scharr':
            dx = kwargs.get('dx', 1)
            dy = kwargs.get('dy', 1)
            self.processed_image = self.scharr_edge(image, dx, dy)
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的边缘检测方法"""
        return ['canny', 'sobel', 'laplacian', 'prewitt', 'roberts', 'scharr']
