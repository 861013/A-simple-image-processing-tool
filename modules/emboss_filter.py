#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浮雕效果模块
实现各种浮雕效果算法
"""

import cv2
import numpy as np
from .base_processor import BaseImageProcessor


class EmbossFilter(BaseImageProcessor):
    """浮雕效果处理类"""
    
    def __init__(self):
        super().__init__()
    
    def basic_emboss(self, image, angle=45):
        """
        基础浮雕效果
        
        Args:
            image (numpy.ndarray): 输入图像
            angle (float): 浮雕角度（度）
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 将角度转换为弧度
        angle_rad = np.radians(angle)
        
        # 创建浮雕核
        kernel = np.array([
            [-2 * np.cos(angle_rad), -np.sin(angle_rad), 0],
            [-np.sin(angle_rad), 1, np.sin(angle_rad)],
            [0, np.sin(angle_rad), 2 * np.cos(angle_rad)]
        ], dtype=np.float32)
        
        # 应用浮雕核
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        if len(image.shape) == 3:
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        else:
            return embossed
    
    def high_emboss(self, image, strength=1.0):
        """
        高浮雕效果
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 浮雕强度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 高浮雕核
        kernel = np.array([
            [-2 * strength, -1 * strength, 0],
            [-1 * strength, 1, 1 * strength],
            [0, 1 * strength, 2 * strength]
        ], dtype=np.float32)
        
        # 应用浮雕核
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        if len(image.shape) == 3:
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        else:
            return embossed
    
    def low_emboss(self, image, strength=0.5):
        """
        低浮雕效果
        
        Args:
            image (numpy.ndarray): 输入图像
            strength (float): 浮雕强度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 低浮雕核
        kernel = np.array([
            [-1 * strength, -0.5 * strength, 0],
            [-0.5 * strength, 1, 0.5 * strength],
            [0, 0.5 * strength, 1 * strength]
        ], dtype=np.float32)
        
        # 应用浮雕核
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        if len(image.shape) == 3:
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        else:
            return embossed
    
    def relief_emboss(self, image, depth=1.0):
        """
        立体浮雕效果
        
        Args:
            image (numpy.ndarray): 输入图像
            depth (float): 浮雕深度
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 立体浮雕核
        kernel = np.array([
            [-depth, -depth, 0],
            [-depth, 1 + 4 * depth, -depth],
            [0, -depth, -depth]
        ], dtype=np.float32)
        
        # 应用浮雕核
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        
        # 转换回RGB
        if len(image.shape) == 3:
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        else:
            return embossed
    
    def process(self, method='basic', **kwargs):
        """
        执行浮雕效果处理
        
        Args:
            method (str): 浮雕方法 ('basic', 'high', 'low', 'relief')
            **kwargs: 方法特定的参数
            
        Returns:
            numpy.ndarray: 处理后的图像
        """
        if not self.has_original_image():
            return None
        
        image = self.original_image.copy()
        
        if method == 'basic':
            angle = kwargs.get('angle', 45)
            self.processed_image = self.basic_emboss(image, angle)
        elif method == 'high':
            strength = kwargs.get('strength', 1.0)
            self.processed_image = self.high_emboss(image, strength)
        elif method == 'low':
            strength = kwargs.get('strength', 0.5)
            self.processed_image = self.low_emboss(image, strength)
        elif method == 'relief':
            depth = kwargs.get('depth', 1.0)
            self.processed_image = self.relief_emboss(image, depth)
        else:
            raise ValueError(f"不支持的浮雕方法: {method}")
        
        return self.processed_image
    
    def get_available_methods(self):
        """获取可用的浮雕方法"""
        return ['basic', 'high', 'low', 'relief']
