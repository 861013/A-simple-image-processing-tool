#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像信息显示模块
显示图像的详细信息，包括格式、尺寸、颜色空间等
"""

import tkinter as tk
from tkinter import ttk


class ImageInfoDisplay:
    """图像信息显示类"""
    
    def __init__(self, parent):
        self.parent = parent
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建滚动文本框
        self.text_widget = tk.Text(self.parent, height=8, width=40, 
                                  font=('Consolas', 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.parent, orient=tk.VERTICAL, 
                                 command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始显示
        self.clear_info()
    
    def update_info(self, image_info, processor=None):
        """
        更新图像信息显示
        
        Args:
            image_info (dict): 图像信息字典
            processor: 图像处理器实例
        """
        if image_info is None:
            self.clear_info()
            return
        
        # 清空现有内容
        self.text_widget.delete(1.0, tk.END)
        
        # 显示基本信息
        info_text = "=== 图像信息 ===\n\n"
        
        # 文件路径
        if 'path' in image_info and image_info['path']:
            info_text += f"文件路径: {image_info['path']}\n"
        
        # 图像尺寸
        if 'width' in image_info and 'height' in image_info:
            info_text += f"图像尺寸: {image_info['width']} × {image_info['height']}\n"
        
        # 通道数
        if 'channels' in image_info:
            info_text += f"通道数: {image_info['channels']}\n"
        
        # 颜色空间
        if 'color_space' in image_info:
            info_text += f"颜色空间: {image_info['color_space']}\n"
        
        # 数据类型
        if 'dtype' in image_info:
            info_text += f"数据类型: {image_info['dtype']}\n"
        
        # 图像大小（字节）
        if 'size' in image_info:
            size_mb = image_info['size'] * image_info['dtype'].itemsize / (1024 * 1024)
            info_text += f"内存大小: {size_mb:.2f} MB\n"
        
        # 图像类型检测
        if processor:
            image_type = processor.detect_image_type()
            info_text += f"图像类型: {image_type}\n"
        
        # 支持的颜色空间
        if processor and hasattr(processor, 'get_available_color_spaces'):
            color_spaces = processor.get_available_color_spaces()
            info_text += f"支持的颜色空间: {', '.join(color_spaces)}\n"
        
        # 图像格式支持
        info_text += "\n=== 支持的格式 ===\n"
        info_text += "输入格式: JPG, PNG, BMP, TIFF, WEBP, GIF\n"
        info_text += "输出格式: JPG, PNG, BMP, TIFF, WEBP, GIF\n"
        
        # 颜色空间支持
        info_text += "\n=== 颜色空间支持 ===\n"
        info_text += "RGB, RGBA, HSV, LAB, YUV, CMYK, 灰度\n"
        
        # 处理功能
        info_text += "\n=== 处理功能 ===\n"
        info_text += "• 平滑滤波 (高斯、均值、中值、双边)\n"
        info_text += "• 锐化滤波 (多种核函数)\n"
        info_text += "• 浮雕效果\n"
        info_text += "• 边缘检测 (Canny、Sobel、Laplacian)\n"
        info_text += "• 色散效果\n"
        info_text += "• 颜色空间转换\n"
        info_text += "• 图像增强\n"
        
        # 插入文本
        self.text_widget.insert(1.0, info_text)
        
        # 使文本只读
        self.text_widget.config(state=tk.DISABLED)
    
    def clear_info(self):
        """清除信息显示"""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        
        default_text = "=== 图像信息 ===\n\n"
        default_text += "请加载图像文件以查看详细信息\n\n"
        default_text += "=== 支持的格式 ===\n"
        default_text += "输入格式: JPG, PNG, BMP, TIFF, WEBP, GIF\n"
        default_text += "输出格式: JPG, PNG, BMP, TIFF, WEBP, GIF\n\n"
        default_text += "=== 颜色空间支持 ===\n"
        default_text += "RGB, RGBA, HSV, LAB, YUV, CMYK, 灰度\n\n"
        default_text += "=== 处理功能 ===\n"
        default_text += "• 平滑滤波 (高斯、均值、中值、双边)\n"
        default_text += "• 锐化滤波 (多种核函数)\n"
        default_text += "• 浮雕效果\n"
        default_text += "• 边缘检测 (Canny、Sobel、Laplacian)\n"
        default_text += "• 色散效果\n"
        default_text += "• 颜色空间转换\n"
        default_text += "• 图像增强"
        
        self.text_widget.insert(1.0, default_text)
        self.text_widget.config(state=tk.DISABLED)