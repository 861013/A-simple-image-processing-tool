#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础图像处理类
提供图像加载、保存等基础功能
支持多种颜色空间和图像格式
支持UTF-8和ASCII编码格式
"""

import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# 可选导入imageio
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# 导入编码管理器
from utils.encoding_manager import safe_print, encoding_manager


class BaseImageProcessor:
    """基础图像处理类"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.color_space = 'RGB'  # 当前颜色空间
        self.image_info = {}  # 图像信息
        
    def load_image(self, image_path):
        """
        加载图像（支持中文路径和多种格式）
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            self.image_path = image_path
            
            # 首先尝试使用PIL加载以支持更多格式
            try:
                pil_image = Image.open(image_path)
                
                # 检测并处理不同的图像模式
                if pil_image.mode == 'RGBA':
                    self.original_image = np.array(pil_image)
                    self.color_space = 'RGBA'
                elif pil_image.mode == 'RGB':
                    self.original_image = np.array(pil_image)
                    self.color_space = 'RGB'
                elif pil_image.mode == 'L':
                    # 灰度图像
                    self.original_image = np.array(pil_image)
                    self.color_space = 'GRAY'
                elif pil_image.mode == 'LA':
                    # 灰度+Alpha
                    self.original_image = np.array(pil_image)
                    self.color_space = 'LA'
                elif pil_image.mode == 'P':
                    # 调色板模式，转换为RGB
                    pil_image = pil_image.convert('RGB')
                    self.original_image = np.array(pil_image)
                    self.color_space = 'RGB'
                elif pil_image.mode == '1':
                    # 黑白模式，转换为灰度
                    pil_image = pil_image.convert('L')
                    self.original_image = np.array(pil_image)
                    self.color_space = 'GRAY'
                elif pil_image.mode == 'CMYK':
                    # CMYK模式，转换为RGB
                    pil_image = pil_image.convert('RGB')
                    self.original_image = np.array(pil_image)
                    self.color_space = 'RGB'
                elif pil_image.mode in ['RGBX', 'RGBa']:
                    # RGB+填充或RGB+预乘Alpha
                    pil_image = pil_image.convert('RGB')
                    self.original_image = np.array(pil_image)
                    self.color_space = 'RGB'
                else:
                    # 其他模式，尝试转换为RGB
                    try:
                        pil_image = pil_image.convert('RGB')
                        self.original_image = np.array(pil_image)
                        self.color_space = 'RGB'
                    except Exception:
                        # 如果转换失败，尝试转换为灰度
                        pil_image = pil_image.convert('L')
                        self.original_image = np.array(pil_image)
                        self.color_space = 'GRAY'
                
                # 更新图像信息
                self._update_image_info()
                return True
                
            except Exception as pil_error:
                print(f"PIL加载失败，尝试OpenCV: {pil_error}")
                
                # 如果PIL失败，使用OpenCV
                img_array = np.fromfile(image_path, dtype=np.uint8)
                
                # 尝试不同的加载模式
                self.original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if self.original_image is None:
                    # 尝试加载为灰度图
                    self.original_image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    if self.original_image is not None:
                        self.color_space = 'GRAY'
                        # 转换为3通道以便后续处理
                        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
                    else:
                        # 尝试加载为RGBA
                        self.original_image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                        if self.original_image is not None:
                            if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 4:
                                # RGBA图像
                                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGRA2RGBA)
                                self.color_space = 'RGBA'
                            elif len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
                                # BGR图像
                                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                                self.color_space = 'RGB'
                            else:
                                # 灰度图像
                                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
                                self.color_space = 'GRAY'
                        else:
                            raise ValueError("无法加载图像文件")
                else:
                    # 转换为RGB格式
                    self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    self.color_space = 'RGB'
                
                # 更新图像信息
                self._update_image_info()
                return True
                
        except Exception as e:
            safe_print(f"加载图像错误: {e}")
            return False
    
    def _update_image_info(self):
        """更新图像信息"""
        if self.original_image is not None:
            self.image_info = {
                'shape': self.original_image.shape,
                'dtype': self.original_image.dtype,
                'size': self.original_image.size,
                'channels': self.original_image.shape[2] if len(self.original_image.shape) == 3 else 1,
                'height': self.original_image.shape[0],
                'width': self.original_image.shape[1],
                'color_space': self.color_space,
                'path': self.image_path
            }
    
    def convert_color_space(self, target_space):
        """
        转换颜色空间
        
        Args:
            target_space (str): 目标颜色空间 ('RGB', 'HSV', 'LAB', 'YUV', 'GRAY')
            
        Returns:
            bool: 转换成功返回True，失败返回False
        """
        if self.original_image is None:
            return False
        
        try:
            if self.color_space == target_space:
                return True
            
            # 定义颜色空间转换映射
            conversions = {
                ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
                ('RGB', 'LAB'): cv2.COLOR_RGB2LAB,
                ('RGB', 'YUV'): cv2.COLOR_RGB2YUV,
                ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
                ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
                ('LAB', 'RGB'): cv2.COLOR_LAB2RGB,
                ('YUV', 'RGB'): cv2.COLOR_YUV2RGB,
                ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
            }
            
            conversion_key = (self.color_space, target_space)
            if conversion_key in conversions:
                if target_space == 'GRAY':
                    # 灰度转换特殊处理
                    gray = cv2.cvtColor(self.original_image, conversions[conversion_key])
                    self.original_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    self.original_image = cv2.cvtColor(self.original_image, conversions[conversion_key])
                
                self.color_space = target_space
                self._update_image_info()
                return True
            else:
                safe_print(f"不支持的颜色空间转换: {self.color_space} -> {target_space}")
                return False
                
        except Exception as e:
            safe_print(f"颜色空间转换错误: {e}")
            return False
    
    def save_image(self, output_path, image=None, quality=95, format=None):
        """
        保存图像（支持中文路径和多种格式）
        
        Args:
            output_path (str): 保存路径
            image (numpy.ndarray, optional): 要保存的图像，默认为处理后的图像
            quality (int): 图像质量 (1-100)
            format (str, optional): 强制指定格式
            
        Returns:
            bool: 保存成功返回True，失败返回False
        """
        try:
            if image is None:
                image = self.processed_image
                if image is None:
                    image = self.original_image
                
            if image is None:
                raise ValueError("没有图像可保存")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir != '':
                os.makedirs(output_dir, exist_ok=True)
            
            # 获取文件扩展名
            ext = os.path.splitext(output_path)[1].lower()
            
            # 如果指定了格式，使用指定格式
            if format:
                ext = f".{format.lower()}"
            
            # 根据扩展名选择保存方法
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']:
                # 使用PIL保存以支持更多格式
                try:
                    if len(image.shape) == 3:
                        # 确保图像是RGB格式
                        if image.shape[2] == 4:  # RGBA
                            pil_image = Image.fromarray(image, 'RGBA')
                        else:  # RGB
                            pil_image = Image.fromarray(image, 'RGB')
                    else:  # 灰度图
                        pil_image = Image.fromarray(image, 'L')
                    
                    # 设置保存参数
                    save_kwargs = {}
                    if ext in ['.jpg', '.jpeg']:
                        save_kwargs['quality'] = quality
                        save_kwargs['optimize'] = True
                    elif ext == '.png':
                        save_kwargs['optimize'] = True
                    elif ext == '.webp':
                        save_kwargs['quality'] = quality
                        save_kwargs['method'] = 6  # 最高压缩
                    elif ext == '.gif':
                        # GIF格式需要特殊处理
                        if len(image.shape) == 3:
                            pil_image = pil_image.convert('P', palette=Image.ADAPTIVE)
                    
                    pil_image.save(output_path, **save_kwargs)
                    return True
                    
                except Exception as pil_error:
                    safe_print(f"PIL保存失败，尝试OpenCV: {pil_error}")
                    
                    # 如果PIL失败，使用OpenCV
                    if len(image.shape) == 3:
                        if image.shape[2] == 4:  # RGBA
                            save_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                        else:  # RGB
                            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        save_image = image
                    
                    # 设置编码参数
                    encode_param = []
                    if ext in ['.jpg', '.jpeg']:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    elif ext == '.png':
                        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                    elif ext == '.webp':
                        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
                    
                    success, encoded_img = cv2.imencode(ext, save_image, encode_param)
                    if success:
                        encoded_img.tofile(output_path)
                        return True
                    else:
                        raise ValueError("图像编码失败")
            else:
                # 使用imageio保存其他格式（如果可用）
                if HAS_IMAGEIO:
                    try:
                        imageio.imwrite(output_path, image)
                        return True
                    except Exception as imageio_error:
                        safe_print(f"imageio保存失败: {imageio_error}")
                        return False
                else:
                    safe_print(f"不支持的文件格式: {ext}，需要安装imageio库")
                    return False
                
        except Exception as e:
            safe_print(f"保存图像错误: {e}")
            return False
    
    def get_image_info(self):
        """
        获取图像信息
        
        Returns:
            dict: 包含图像信息的字典
        """
        return self.image_info.copy() if self.image_info else None
    
    def reset_processed_image(self):
        """重置处理后的图像"""
        self.processed_image = None
    
    def has_original_image(self):
        """检查是否有原始图像"""
        return self.original_image is not None
    
    def has_processed_image(self):
        """检查是否有处理后的图像"""
        return self.processed_image is not None
    
    def is_grayscale(self):
        """检查是否为灰度图像"""
        return self.color_space == 'GRAY'
    
    def is_color(self):
        """检查是否为彩色图像"""
        return self.color_space in ['RGB', 'RGBA', 'HSV', 'LAB', 'YUV']
    
    def get_available_color_spaces(self):
        """获取可用的颜色空间列表"""
        return ['RGB', 'HSV', 'LAB', 'YUV', 'GRAY']
    
    def detect_image_type(self):
        """检测图像类型"""
        if self.original_image is None:
            return "未知"
        
        if len(self.original_image.shape) == 2:
            return "灰度图像"
        elif len(self.original_image.shape) == 3:
            if self.original_image.shape[2] == 3:
                return "彩色图像 (RGB)"
            elif self.original_image.shape[2] == 4:
                return "彩色图像 (RGBA)"
            else:
                return f"多通道图像 ({self.original_image.shape[2]}通道)"
        else:
            return "未知格式"
