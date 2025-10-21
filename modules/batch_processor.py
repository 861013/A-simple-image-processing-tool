#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理模块
支持批量处理多张图片
"""

import os
import cv2
import numpy as np
from datetime import datetime
from .base_processor import BaseImageProcessor
from modules.smooth_filter import SmoothFilter
from modules.sharpen_filter import SharpenFilter
from modules.emboss_filter import EmbossFilter
from modules.edge_detection import EdgeDetection
from modules.chromatic_aberration import ChromaticAberration
from modules.color_conversion import ColorConversion


class BatchProcessor:
    """批量处理类"""
    
    def __init__(self):
        self.processors = {
            'smooth': SmoothFilter(),
            'sharpen': SharpenFilter(),
            'emboss': EmbossFilter(),
            'edge': EdgeDetection(),
            'chromatic': ChromaticAberration(),
            'color_conversion': ColorConversion()
        }
        self.output_dir = "batch_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_supported_formats(self):
        """获取支持的图像格式"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def scan_directory(self, directory_path):
        """
        扫描目录中的图像文件
        
        Args:
            directory_path (str): 目录路径
            
        Returns:
            list: 图像文件路径列表
        """
        image_files = []
        supported_formats = self.get_supported_formats()
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename.lower())
                    if ext in supported_formats:
                        image_files.append(file_path)
        except Exception as e:
            print(f"扫描目录错误: {e}")
        
        return sorted(image_files)
    
    def process_single_image(self, image_path, processor_type, method, **kwargs):
        """
        处理单张图像
        
        Args:
            image_path (str): 图像路径
            processor_type (str): 处理器类型
            method (str): 处理方法
            **kwargs: 处理参数
            
        Returns:
            tuple: (成功标志, 输出路径, 错误信息)
        """
        try:
            processor = self.processors[processor_type]
            
            # 加载图像
            if not processor.load_image(image_path):
                return False, None, "无法加载图像"
            
            # 处理图像
            processed_image = processor.process(method=method, **kwargs)
            if processed_image is None:
                return False, None, "图像处理失败"
            
            # 生成输出文件名
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{name}_{processor_type}_{method}_{timestamp}{ext}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 保存图像
            if processor.save_image(output_path, processed_image):
                return True, output_path, None
            else:
                return False, None, "保存图像失败"
                
        except Exception as e:
            return False, None, str(e)
    
    def process_batch(self, image_paths, processor_type, method, **kwargs):
        """
        批量处理图像
        
        Args:
            image_paths (list): 图像路径列表
            processor_type (str): 处理器类型
            method (str): 处理方法
            **kwargs: 处理参数
            
        Returns:
            dict: 处理结果统计
        """
        results = {
            'total': len(image_paths),
            'success': 0,
            'failed': 0,
            'success_files': [],
            'failed_files': [],
            'errors': []
        }
        
        print(f"开始批量处理 {len(image_paths)} 张图像...")
        print(f"处理器: {processor_type}, 方法: {method}")
        print("-" * 50)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"处理第 {i}/{len(image_paths)} 张: {os.path.basename(image_path)}")
            
            success, output_path, error = self.process_single_image(
                image_path, processor_type, method, **kwargs
            )
            
            if success:
                results['success'] += 1
                results['success_files'].append({
                    'input': image_path,
                    'output': output_path
                })
                print(f"  ✓ 成功: {os.path.basename(output_path)}")
            else:
                results['failed'] += 1
                results['failed_files'].append(image_path)
                results['errors'].append({
                    'file': image_path,
                    'error': error
                })
                print(f"  ✗ 失败: {error}")
        
        print("-" * 50)
        print(f"批量处理完成!")
        print(f"成功: {results['success']}/{results['total']}")
        print(f"失败: {results['failed']}/{results['total']}")
        
        return results
    
    def process_directory(self, directory_path, processor_type, method, **kwargs):
        """
        处理目录中的所有图像
        
        Args:
            directory_path (str): 目录路径
            processor_type (str): 处理器类型
            method (str): 处理方法
            **kwargs: 处理参数
            
        Returns:
            dict: 处理结果统计
        """
        image_files = self.scan_directory(directory_path)
        
        if not image_files:
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'success_files': [],
                'failed_files': [],
                'errors': [{'file': directory_path, 'error': '未找到支持的图像文件'}]
            }
        
        return self.process_batch(image_files, processor_type, method, **kwargs)
    
    def get_processor_info(self, processor_type):
        """
        获取处理器信息
        
        Args:
            processor_type (str): 处理器类型
            
        Returns:
            dict: 处理器信息
        """
        if processor_type not in self.processors:
            return None
        
        processor = self.processors[processor_type]
        info = {
            'name': processor_type,
            'available_methods': processor.get_available_methods() if hasattr(processor, 'get_available_methods') else [],
            'supports_color': processor.is_color() if hasattr(processor, 'is_color') else True,
            'supports_grayscale': processor.is_grayscale() if hasattr(processor, 'is_grayscale') else True
        }
        
        return info
    
    def get_all_processors_info(self):
        """获取所有处理器信息"""
        processors_info = {}
        for processor_type in self.processors.keys():
            processors_info[processor_type] = self.get_processor_info(processor_type)
        return processors_info
