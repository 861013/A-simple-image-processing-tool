#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制面板模块
负责用户界面控制和参数设置
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os


class ControlPanel:
    """控制面板类"""
    
    def __init__(self, parent_frame, callback_functions=None):
        self.parent_frame = parent_frame
        self.callbacks = callback_functions or {}
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 文件操作框架
        file_frame = ttk.LabelFrame(self.parent_frame, text="文件操作", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件操作按钮
        ttk.Button(file_frame, text="选择图像", 
                  command=self.load_image).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="保存图像", 
                  command=self.save_image).pack(fill=tk.X)
        
        # 分隔线
        ttk.Separator(self.parent_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 图像处理框架
        process_frame = ttk.LabelFrame(self.parent_frame, text="图像处理", padding="10")
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理算法选择
        self.process_var = tk.StringVar(value="smooth")
        algorithms = [
            ("平滑滤波", "smooth"),
            ("锐化滤波", "sharpen"),
            ("浮雕效果", "emboss"),
            ("边缘检测", "edge"),
            ("色散效果", "chromatic"),
            ("颜色转换", "color_conversion")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(process_frame, text=text, variable=self.process_var, 
                           value=value, command=self.update_parameters).pack(anchor=tk.W)
        
        # 参数控制框架
        self.params_frame = ttk.Frame(self.parent_frame)
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理按钮框架
        button_frame = ttk.Frame(self.parent_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 处理按钮
        ttk.Button(button_frame, text="开始处理", 
                  command=self.process_image).pack(fill=tk.X, pady=(0, 5))
        
        # 重置按钮
        ttk.Button(button_frame, text="重置图像", 
                  command=self.reset_image).pack(fill=tk.X)
        
        # 初始化参数控制
        self.update_parameters()
        
    def update_parameters(self):
        """更新参数控制界面"""
        # 清除现有参数控件
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        method = self.process_var.get()
        
        if method == "smooth":
            self.create_smooth_parameters()
        elif method == "sharpen":
            self.create_sharpen_parameters()
        elif method == "emboss":
            self.create_emboss_parameters()
        elif method == "edge":
            self.create_edge_parameters()
        elif method == "chromatic":
            self.create_chromatic_parameters()
        elif method == "color_conversion":
            self.create_color_conversion_parameters()
        
        # 确保按钮始终存在
        self.create_action_buttons()
    
    def create_action_buttons(self):
        """创建操作按钮"""
        # 分隔线
        ttk.Separator(self.params_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 5))
        
        # 处理按钮
        ttk.Button(self.params_frame, text="开始处理", 
                  command=self.process_image).pack(fill=tk.X, pady=(0, 5))
        
        # 重置按钮
        ttk.Button(self.params_frame, text="重置图像", 
                  command=self.reset_image).pack(fill=tk.X)
    
    def create_smooth_parameters(self):
        """创建平滑滤波参数控件"""
        ttk.Label(self.params_frame, text="滤波方法:").pack(anchor=tk.W)
        self.smooth_method_var = tk.StringVar(value="gaussian")
        smooth_combo = ttk.Combobox(self.params_frame, textvariable=self.smooth_method_var,
                                   values=["gaussian", "mean", "median", "bilateral"], 
                                   state="readonly", width=15)
        smooth_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="核大小:").pack(anchor=tk.W)
        self.kernel_size_var = tk.IntVar(value=5)
        kernel_scale = ttk.Scale(self.params_frame, from_=3, to=15, 
                                variable=self.kernel_size_var, orient=tk.HORIZONTAL)
        kernel_scale.pack(fill=tk.X, pady=(0, 5))
    
    def create_sharpen_parameters(self):
        """创建锐化滤波参数控件"""
        ttk.Label(self.params_frame, text="锐化方法:").pack(anchor=tk.W)
        self.sharpen_method_var = tk.StringVar(value="unsharp_mask")
        sharpen_combo = ttk.Combobox(self.params_frame, textvariable=self.sharpen_method_var,
                                    values=["unsharp_mask (反锐化掩模)", "laplacian (拉普拉斯)", "high_pass (高通)", "sobel (索贝尔)"], 
                                    state="readonly", width=20)
        sharpen_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="锐化强度:").pack(anchor=tk.W)
        self.strength_var = tk.DoubleVar(value=1.0)
        strength_scale = ttk.Scale(self.params_frame, from_=0.1, to=3.0, 
                                  variable=self.strength_var, orient=tk.HORIZONTAL)
        strength_scale.pack(fill=tk.X, pady=(0, 5))
    
    def create_emboss_parameters(self):
        """创建浮雕效果参数控件"""
        ttk.Label(self.params_frame, text="浮雕方法:").pack(anchor=tk.W)
        self.emboss_method_var = tk.StringVar(value="basic")
        emboss_combo = ttk.Combobox(self.params_frame, textvariable=self.emboss_method_var,
                                   values=["basic (基础)", "high (高浮雕)", "low (低浮雕)", "relief (立体浮雕)"], 
                                   state="readonly", width=20)
        emboss_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="浮雕强度:").pack(anchor=tk.W)
        self.emboss_strength_var = tk.DoubleVar(value=1.0)
        emboss_scale = ttk.Scale(self.params_frame, from_=0.1, to=3.0, 
                                variable=self.emboss_strength_var, orient=tk.HORIZONTAL)
        emboss_scale.pack(fill=tk.X, pady=(0, 5))
    
    def create_edge_parameters(self):
        """创建边缘检测参数控件"""
        ttk.Label(self.params_frame, text="检测方法:").pack(anchor=tk.W)
        self.edge_method_var = tk.StringVar(value="canny")
        edge_combo = ttk.Combobox(self.params_frame, textvariable=self.edge_method_var,
                                 values=["canny (坎尼)", "sobel (索贝尔)", "laplacian (拉普拉斯)", "prewitt (普瑞维特)", "roberts (罗伯茨)", "scharr (沙尔)"], 
                                 state="readonly", width=20)
        edge_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Canny参数
        ttk.Label(self.params_frame, text="低阈值:").pack(anchor=tk.W)
        self.low_threshold_var = tk.IntVar(value=50)
        low_scale = ttk.Scale(self.params_frame, from_=10, to=200, 
                             variable=self.low_threshold_var, orient=tk.HORIZONTAL)
        low_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="高阈值:").pack(anchor=tk.W)
        self.high_threshold_var = tk.IntVar(value=150)
        high_scale = ttk.Scale(self.params_frame, from_=50, to=300, 
                              variable=self.high_threshold_var, orient=tk.HORIZONTAL)
        high_scale.pack(fill=tk.X, pady=(0, 5))
    
    def create_chromatic_parameters(self):
        """创建色散效果参数控件"""
        ttk.Label(self.params_frame, text="色散方法:").pack(anchor=tk.W)
        self.chromatic_method_var = tk.StringVar(value="basic")
        chromatic_combo = ttk.Combobox(self.params_frame, textvariable=self.chromatic_method_var,
                                      values=["basic (基础)", "advanced (高级)", "radial (径向)", "prism (棱镜)"], 
                                      state="readonly", width=20)
        chromatic_combo.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="色散强度:").pack(anchor=tk.W)
        self.intensity_var = tk.IntVar(value=5)
        intensity_scale = ttk.Scale(self.params_frame, from_=1, to=20, 
                                   variable=self.intensity_var, orient=tk.HORIZONTAL)
        intensity_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="色散方向:").pack(anchor=tk.W)
        self.direction_var = tk.StringVar(value="horizontal")
        direction_combo = ttk.Combobox(self.params_frame, textvariable=self.direction_var,
                                      values=["horizontal (水平)", "vertical (垂直)", "radial (径向)"], 
                                      state="readonly", width=20)
        direction_combo.pack(fill=tk.X, pady=(0, 5))
    
    def create_color_conversion_parameters(self):
        """创建颜色转换参数控件"""
        ttk.Label(self.params_frame, text="转换方法:").pack(anchor=tk.W)
        self.color_method_var = tk.StringVar(value="convert_color_space")
        color_combo = ttk.Combobox(self.params_frame, textvariable=self.color_method_var,
                                  values=["convert_color_space (颜色空间转换)", "convert_to_grayscale (转灰度)", "enhance_color (颜色增强)", "convert_to_rgb (转RGB)"], 
                                  state="readonly", width=25)
        color_combo.pack(fill=tk.X, pady=(0, 5))
        
        # 灰度转换方法
        ttk.Label(self.params_frame, text="灰度方法:").pack(anchor=tk.W)
        self.gray_method_var = tk.StringVar(value="luminance")
        gray_combo = ttk.Combobox(self.params_frame, textvariable=self.gray_method_var,
                                 values=["luminance (亮度)", "average (平均)", "max (最大值)", "min (最小值)"], 
                                 state="readonly", width=20)
        gray_combo.pack(fill=tk.X, pady=(0, 5))
        
        # 颜色增强参数
        ttk.Label(self.params_frame, text="增强因子:").pack(anchor=tk.W)
        self.enhancement_factor_var = tk.DoubleVar(value=1.2)
        enhancement_scale = ttk.Scale(self.params_frame, from_=0.5, to=2.0, 
                                    variable=self.enhancement_factor_var, orient=tk.HORIZONTAL)
        enhancement_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(self.params_frame, text="颜色空间:").pack(anchor=tk.W)
        self.color_space_var = tk.StringVar(value="HSV")
        color_space_combo = ttk.Combobox(self.params_frame, textvariable=self.color_space_var,
                                       values=["HSV (色调饱和度明度)", "LAB (感知均匀)", "YUV (亮度色度)", "GRAY (灰度)"], 
                                       state="readonly", width=20)
        color_space_combo.pack(fill=tk.X, pady=(0, 5))
    
    def extract_english_name(self, display_name):
        """从显示名称中提取英文名称"""
        if '(' in display_name:
            return display_name.split('(')[0].strip()
        return display_name
    
    def get_processing_parameters(self):
        """获取处理参数"""
        method = self.process_var.get()
        params = {'method': method}
        
        if method == "smooth":
            params.update({
                'filter_method': self.smooth_method_var.get(),
                'kernel_size': self.kernel_size_var.get()
            })
        elif method == "sharpen":
            params.update({
                'filter_method': self.extract_english_name(self.sharpen_method_var.get()),
                'strength': self.strength_var.get()
            })
        elif method == "emboss":
            params.update({
                'filter_method': self.extract_english_name(self.emboss_method_var.get()),
                'strength': self.emboss_strength_var.get()
            })
        elif method == "edge":
            params.update({
                'filter_method': self.extract_english_name(self.edge_method_var.get()),
                'low_threshold': self.low_threshold_var.get(),
                'high_threshold': self.high_threshold_var.get()
            })
        elif method == "chromatic":
            params.update({
                'filter_method': self.extract_english_name(self.chromatic_method_var.get()),
                'intensity': self.intensity_var.get(),
                'direction': self.extract_english_name(self.direction_var.get())
            })
        elif method == "color_conversion":
            params.update({
                'filter_method': self.extract_english_name(self.color_method_var.get()),
                'gray_method': self.extract_english_name(self.gray_method_var.get()),
                'enhancement_factor': self.enhancement_factor_var.get(),
                'color_space': self.extract_english_name(self.color_space_var.get())
            })
        
        return params
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("所有支持的图像", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.gif"),
                ("JPEG图像", "*.jpg *.jpeg"),
                ("PNG图像", "*.png"),
                ("BMP图像", "*.bmp"),
                ("TIFF图像", "*.tiff *.tif"),
                ("WebP图像", "*.webp"),
                ("GIF图像", "*.gif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path and 'load_image' in self.callbacks:
            self.callbacks['load_image'](file_path)
    
    def save_image(self):
        """保存图像"""
        if 'save_image' in self.callbacks:
            self.callbacks['save_image']()
    
    def process_image(self):
        """处理图像"""
        if 'process_image' in self.callbacks:
            self.callbacks['process_image']()
    
    def reset_image(self):
        """重置图像"""
        if 'reset_image' in self.callbacks:
            self.callbacks['reset_image']()
