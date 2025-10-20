#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理功能演示脚本
展示程序对各种图像格式和颜色空间的支持
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.base_processor import BaseImageProcessor
from modules.color_conversion import ColorConversion
from modules.smooth_filter import SmoothFilter


def demo_image_formats():
    """演示图像格式支持"""
    print("=== 图像格式支持演示 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    processor = BaseImageProcessor()
    processor.processed_image = test_image
    
    # 测试不同格式保存
    formats = {
        'jpg': 'JPEG格式',
        'png': 'PNG格式（支持透明）',
        'bmp': 'BMP格式',
        'webp': 'WebP格式（现代压缩）'
    }
    
    print("支持的输出格式：")
    for ext, desc in formats.items():
        filename = f"demo_output.{ext}"
        if processor.save_image(filename):
            file_size = os.path.getsize(filename)
            print(f"  [OK] {desc}: {filename} ({file_size} 字节)")
            os.remove(filename)  # 清理
        else:
            print(f"  [FAIL] {desc}: 保存失败")


def demo_color_spaces():
    """演示颜色空间转换"""
    print("\n=== 颜色空间转换演示 ===")
    
    # 创建彩色测试图像
    color_image = np.zeros((50, 50, 3), dtype=np.uint8)
    color_image[:, :, 0] = 255  # 红色
    color_image[:, :, 1] = 128  # 绿色
    color_image[:, :, 2] = 64   # 蓝色
    
    converter = ColorConversion()
    converter.original_image = color_image
    
    # 测试各种颜色空间转换
    conversions = [
        ('RGB', 'HSV'),
        ('RGB', 'LAB'),
        ('RGB', 'YUV'),
        ('RGB', 'GRAY')
    ]
    
    print("颜色空间转换：")
    for from_space, to_space in conversions:
        try:
            if to_space == 'GRAY':
                result = converter.convert_to_grayscale(color_image, 'luminance')
            else:
                result = converter.convert_color_space(to_space)
            
            if result is not None:
                print(f"  [OK] {from_space} → {to_space}: 成功")
            else:
                print(f"  [FAIL] {from_space} → {to_space}: 失败")
        except Exception as e:
            print(f"  [ERROR] {from_space} → {to_space}: 错误 - {e}")


def demo_image_processing():
    """演示图像处理功能"""
    print("\n=== 图像处理功能演示 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试平滑滤波
    smooth_processor = SmoothFilter()
    smooth_processor.original_image = test_image
    
    filters = [
        ('gaussian', '高斯模糊'),
        ('mean', '均值滤波'),
        ('median', '中值滤波'),
        ('bilateral', '双边滤波')
    ]
    
    print("平滑滤波算法：")
    for method, name in filters:
        try:
            result = smooth_processor.process(method)
            if result is not None:
                print(f"  [OK] {name}: 处理成功")
            else:
                print(f"  [FAIL] {name}: 处理失败")
        except Exception as e:
            print(f"  [ERROR] {name}: 错误 - {e}")


def demo_chinese_support():
    """演示中文路径支持"""
    print("\n=== 中文路径支持演示 ===")
    
    # 创建中文目录
    chinese_dir = "演示目录"
    os.makedirs(chinese_dir, exist_ok=True)
    
    processor = BaseImageProcessor()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processor.processed_image = test_image
    
    chinese_filename = os.path.join(chinese_dir, "测试图像.png")
    
    try:
        # 保存到中文路径
        if processor.save_image(chinese_filename):
            print(f"  [OK] 中文路径保存成功: {chinese_filename}")
            
            # 从中文路径加载
            if processor.load_image(chinese_filename):
                print(f"  [OK] 中文路径加载成功")
                info = processor.get_image_info()
                print(f"    - 图像尺寸: {info['width']} × {info['height']}")
                print(f"    - 颜色空间: {info['color_space']}")
            else:
                print(f"  [FAIL] 中文路径加载失败")
        else:
            print(f"  [FAIL] 中文路径保存失败")
    except Exception as e:
        print(f"  [ERROR] 中文路径错误: {e}")
    finally:
        # 清理
        import shutil
        if os.path.exists(chinese_dir):
            shutil.rmtree(chinese_dir)


def demo_image_info():
    """演示图像信息检测"""
    print("\n=== 图像信息检测演示 ===")
    
    processor = BaseImageProcessor()
    
    # 创建不同类型的测试图像
    test_images = [
        (np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8), "RGB彩色图像"),
        (np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8), "RGBA图像"),
        (np.random.randint(0, 255, (50, 50), dtype=np.uint8), "灰度图像"),
    ]
    
    print("图像类型检测：")
    for image, description in test_images:
        processor.original_image = image
        image_type = processor.detect_image_type()
        print(f"  [OK] {description}: {image_type}")


def main():
    """主演示函数"""
    print("=" * 60)
    print("图像增强处理工具 - 功能演示")
    print("=" * 60)
    
    try:
        demo_image_formats()
        demo_color_spaces()
        demo_image_processing()
        demo_chinese_support()
        demo_image_info()
        
        print("\n" + "=" * 60)
        print("演示完成！程序支持多种图像格式和颜色空间处理。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")


if __name__ == "__main__":
    main()
