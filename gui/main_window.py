#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口模块
整合所有GUI组件
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from datetime import datetime

from .image_display import ImageDisplay
from .control_panel import ControlPanel
from .image_info_display import ImageInfoDisplay
from modules.smooth_filter import SmoothFilter
from modules.sharpen_filter import SharpenFilter
from modules.emboss_filter import EmbossFilter
from modules.edge_detection import EdgeDetection
from modules.chromatic_aberration import ChromaticAberration
from modules.color_conversion import ColorConversion


class MainWindow:
    """主窗口类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强处理工具 - 模块化版本")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 创建输出目录
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化图像处理器
        self.image_processors = {
            'smooth': SmoothFilter(),
            'sharpen': SharpenFilter(),
            'emboss': EmbossFilter(),
            'edge': EdgeDetection(),
            'chromatic': ChromaticAberration(),
            'color_conversion': ColorConversion()
        }
        
        self.current_processor = None
        self.original_image = None
        self.processed_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像增强处理工具", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 创建控制面板
        self.control_panel = ControlPanel(control_frame, {
            'load_image': self.load_image,
            'save_image': self.save_image,
            'process_image': self.process_image,
            'reset_image': self.reset_image
        })
        
        # 图像信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="图像信息", padding="5")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        
        # 创建图像信息显示组件
        self.image_info_display = ImageInfoDisplay(info_frame)
        
        # 图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="5")
        image_frame.grid(row=1, column=1, columnspan=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建图像显示组件
        self.image_display = ImageDisplay(image_frame)
        
    def load_image(self, file_path):
        """加载图像"""
        try:
            # 尝试所有处理器加载图像
            success = False
            for processor in self.image_processors.values():
                if processor.load_image(file_path):
                    success = True
                    self.original_image = processor.original_image
                    self.current_processor = processor  # 设置当前处理器
                    break
            
            if success:
                self.image_display.update_original_image(self.original_image)
                # 更新图像信息显示
                image_info = self.current_processor.get_image_info()
                self.image_info_display.update_info(image_info, self.current_processor)
                messagebox.showinfo("成功", "图像加载成功！")
            else:
                messagebox.showerror("错误", "无法加载图像文件！")
        except Exception as e:
            messagebox.showerror("错误", f"加载图像时发生错误：{str(e)}")
    
    def process_image(self):
        """处理图像"""
        try:
            if self.original_image is None:
                messagebox.showwarning("警告", "请先选择图像文件！")
                return
            
            # 获取处理参数
            params = self.control_panel.get_processing_parameters()
            method = params['method']
            
            # 选择对应的处理器
            self.current_processor = self.image_processors[method]
            
            # 确保处理器加载了相同的图像
            if self.current_processor.original_image is None:
                self.current_processor.original_image = self.original_image.copy()
            
            # 准备处理器参数
            processor_params = {}
            if method == 'smooth':
                processor_params = {
                    'method': params['filter_method'],
                    'kernel_size': params['kernel_size']
                }
            elif method == 'sharpen':
                processor_params = {
                    'method': params['filter_method'],
                    'strength': params['strength']
                }
                # 为unsharp_mask方法添加额外参数
                if params['filter_method'] == 'unsharp_mask':
                    processor_params.update({
                        'radius': 1.0,
                        'threshold': 0
                    })
            elif method == 'emboss':
                processor_params = {
                    'method': params['filter_method']
                }
                # 为不同浮雕方法添加相应参数
                if params['filter_method'] == 'basic':
                    processor_params['angle'] = 45
                elif params['filter_method'] == 'relief':
                    processor_params['depth'] = params['strength']
                elif params['filter_method'] in ['high', 'low']:
                    processor_params['strength'] = params['strength']
            elif method == 'edge':
                processor_params = {
                    'method': params['filter_method']
                }
                # 为Canny方法添加阈值参数
                if params['filter_method'] == 'canny':
                    processor_params.update({
                        'low_threshold': params['low_threshold'],
                        'high_threshold': params['high_threshold'],
                        'aperture_size': 3
                    })
                # 为Sobel方法添加参数
                elif params['filter_method'] == 'sobel':
                    processor_params.update({
                        'dx': 1,
                        'dy': 1,
                        'ksize': 3
                    })
                # 为Laplacian方法添加参数
                elif params['filter_method'] == 'laplacian':
                    processor_params['ksize'] = 3
            elif method == 'chromatic':
                processor_params = {
                    'method': params['filter_method'],
                    'intensity': params['intensity']
                }
                # 为基础色散方法添加方向参数
                if params['filter_method'] == 'basic':
                    processor_params['direction'] = params['direction']
                # 为高级色散方法添加模糊参数
                elif params['filter_method'] == 'advanced':
                    processor_params['blur_radius'] = 1
                # 为径向色散方法添加中心参数
                elif params['filter_method'] == 'radial':
                    processor_params['center'] = None
                # 为棱镜效果添加角度参数
                elif params['filter_method'] == 'prism':
                    processor_params['angle'] = 0
            elif method == 'color_conversion':
                processor_params = {
                    'method': params['filter_method']
                }
                # 根据不同的转换方法添加相应参数
                if params['filter_method'] == 'convert_to_grayscale':
                    processor_params['gray_method'] = params.get('gray_method', 'luminance')
                elif params['filter_method'] == 'enhance_color':
                    processor_params['enhancement_factor'] = params.get('enhancement_factor', 1.2)
                elif params['filter_method'] == 'convert_color_space':
                    processor_params['color_space'] = params.get('color_space', 'HSV')
                elif params['filter_method'] == 'convert_to_rgb':
                    processor_params['from_space'] = 'RGB'
            
            # 处理图像
            self.processed_image = self.current_processor.process(**processor_params)
            
            if self.processed_image is not None:
                self.image_display.update_processed_image(self.processed_image)
                # 更新图像信息显示
                image_info = self.current_processor.get_image_info()
                self.image_info_display.update_info(image_info, self.current_processor)
                messagebox.showinfo("成功", "图像处理完成！")
            else:
                messagebox.showerror("错误", "图像处理失败！")
                
        except Exception as e:
            messagebox.showerror("错误", f"处理图像时发生错误：{str(e)}")
    
    def save_image(self):
        """保存图像"""
        try:
            if self.processed_image is None:
                messagebox.showwarning("警告", "没有处理后的图像可保存！")
                return
            
            # 选择保存格式
            file_path = filedialog.asksaveasfilename(
                title="保存图像",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG图像", "*.jpg"),
                    ("PNG图像", "*.png"),
                    ("BMP图像", "*.bmp"),
                    ("TIFF图像", "*.tiff"),
                    ("WebP图像", "*.webp"),
                    ("GIF图像", "*.gif"),
                    ("所有文件", "*.*")
                ]
            )
            
            if file_path:
                # 保存图像
                if self.current_processor and self.current_processor.save_image(file_path):
                    messagebox.showinfo("成功", f"图像已保存到: {file_path}")
                else:
                    messagebox.showerror("错误", "保存图像失败！")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存图像时发生错误：{str(e)}")
    
    def reset_image(self):
        """重置图像"""
        try:
            self.processed_image = None
            self.current_processor = None
            
            # 重置所有处理器
            for processor in self.image_processors.values():
                processor.reset_processed_image()
            
            self.image_display.update_processed_image(None)
            # 清除图像信息显示
            self.image_info_display.clear_info()
            messagebox.showinfo("成功", "图像已重置！")
            
        except Exception as e:
            messagebox.showerror("错误", f"重置图像时发生错误：{str(e)}")
    
    def run(self):
        """运行主窗口"""
        self.root.mainloop()
