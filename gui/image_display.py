#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像显示模块
负责图像的显示和对比功能
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class ImageDisplay:
    """图像显示类"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.setup_display()
        
    def setup_display(self):
        """设置显示区域"""
        # 创建matplotlib图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title("原始图像", fontsize=12, fontweight='bold')
        self.ax2.set_title("处理后图像", fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # 设置图形样式
        self.fig.patch.set_facecolor('white')
        self.fig.tight_layout(pad=2.0)
        
        # 将matplotlib图形嵌入到tkinter中
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化显示
        self.clear_display()
        
    def display_images(self, original_image=None, processed_image=None):
        """
        显示图像
        
        Args:
            original_image (numpy.ndarray, optional): 原始图像
            processed_image (numpy.ndarray, optional): 处理后的图像
        """
        # 清除之前的显示
        self.ax1.clear()
        self.ax2.clear()
        
        # 显示原始图像
        if original_image is not None:
            self.ax1.imshow(original_image)
            self.ax1.set_title("原始图像", fontsize=12, fontweight='bold')
        else:
            self.ax1.text(0.5, 0.5, '请选择图像文件', 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=14, color='gray')
            self.ax1.set_title("原始图像", fontsize=12, fontweight='bold')
        
        # 显示处理后的图像
        if processed_image is not None:
            self.ax2.imshow(processed_image)
            self.ax2.set_title("处理后图像", fontsize=12, fontweight='bold')
        else:
            self.ax2.text(0.5, 0.5, '请选择处理算法', 
                         ha='center', va='center', transform=self.ax2.transAxes,
                         fontsize=14, color='gray')
            self.ax2.set_title("处理后图像", fontsize=12, fontweight='bold')
        
        # 设置坐标轴
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # 刷新显示
        self.canvas.draw()
        
    def clear_display(self):
        """清除显示"""
        self.display_images()
        
    def update_original_image(self, image):
        """
        更新原始图像显示
        
        Args:
            image (numpy.ndarray): 原始图像
        """
        self.ax1.clear()
        if image is not None:
            self.ax1.imshow(image)
        else:
            self.ax1.text(0.5, 0.5, '请选择图像文件', 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=14, color='gray')
        self.ax1.set_title("原始图像", fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        self.canvas.draw()
        
    def update_processed_image(self, image):
        """
        更新处理后图像显示
        
        Args:
            image (numpy.ndarray): 处理后的图像
        """
        self.ax2.clear()
        if image is not None:
            self.ax2.imshow(image)
        else:
            self.ax2.text(0.5, 0.5, '请选择处理算法', 
                         ha='center', va='center', transform=self.ax2.transAxes,
                         fontsize=14, color='gray')
        self.ax2.set_title("处理后图像", fontsize=12, fontweight='bold')
        self.ax2.axis('off')
        self.canvas.draw()
        
    def get_figure(self):
        """获取matplotlib图形对象"""
        return self.fig
