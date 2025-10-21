#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像格式支持测试脚本
测试程序对各种图像格式和颜色空间的支持
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


def create_test_images():
    """创建各种格式的测试图像"""
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建基础测试图像
    width, height = 200, 200
    
    # RGB图像
    rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_dir, "test_rgb.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    # RGBA图像
    rgba_image = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
    rgba_image[:, :, 3] = 255  # 设置alpha通道
    pil_rgba = Image.fromarray(rgba_image, 'RGBA')
    pil_rgba.save(os.path.join(test_dir, "test_rgba.png"))
    
    # 灰度图像
    gray_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_dir, "test_gray.jpg"), gray_image)
    
    # HSV图像
    hsv_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    hsv_image[:, :, 0] = np.random.randint(0, 180, (height, width))  # H通道范围0-179
    cv2.imwrite(os.path.join(test_dir, "test_hsv.jpg"), cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))
    
    # LAB图像
    lab_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_dir, "test_lab.jpg"), cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR))
    
    # 黑白图像
    bw_image = np.random.randint(0, 2, (height, width), dtype=np.uint8) * 255
    pil_bw = Image.fromarray(bw_image, '1')
    pil_bw.save(os.path.join(test_dir, "test_bw.png"))
    
    # 调色板图像
    palette_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    pil_palette = Image.fromarray(palette_image, 'P')
    pil_palette.save(os.path.join(test_dir, "test_palette.png"))
    
    print(f"测试图像已创建到 {test_dir} 目录")
    return test_dir


def test_image_loading():
    """测试图像加载功能"""
    print("\n=== 测试图像加载功能 ===")
    
    test_dir = create_test_images()
    processor = BaseImageProcessor()
    
    test_files = [
        "test_rgb.jpg",
        "test_rgba.png", 
        "test_gray.jpg",
        "test_hsv.jpg",
        "test_lab.jpg",
        "test_bw.png",
        "test_palette.png"
    ]
    
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        print(f"\n测试文件: {filename}")
        
        if processor.load_image(filepath):
            info = processor.get_image_info()
            print(f"  [OK] 加载成功")
            print(f"  - 尺寸: {info['width']} × {info['height']}")
            print(f"  - 通道数: {info['channels']}")
            print(f"  - 颜色空间: {info['color_space']}")
            print(f"  - 数据类型: {info['dtype']}")
        else:
            print(f"  [FAIL] 加载失败")


def test_color_conversion():
    """测试颜色空间转换功能"""
    print("\n=== 测试颜色空间转换功能 ===")
    
    processor = ColorConversion()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processor.original_image = test_image
    
    # 测试各种转换
    conversions = [
        ('convert_to_rgb', {'from_space': 'RGB'}),
        ('convert_to_grayscale', {'gray_method': 'luminance'}),
        ('enhance_color', {'enhancement_factor': 1.5}),
        ('convert_color_space', {'color_space': 'HSV'}),
        ('convert_color_space', {'color_space': 'LAB'}),
        ('convert_color_space', {'color_space': 'YUV'}),
    ]
    
    for method, params in conversions:
        print(f"\n测试方法: {method}")
        try:
            result = processor.process(method, **params)
            if result is not None:
                print(f"  [OK] 转换成功，输出形状: {result.shape}")
            else:
                print(f"  [FAIL] 转换失败")
        except Exception as e:
            print(f"  [ERROR] 转换错误: {e}")


def test_image_saving():
    """测试图像保存功能"""
    print("\n=== 测试图像保存功能 ===")
    
    processor = BaseImageProcessor()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processor.processed_image = test_image
    
    # 测试各种格式保存
    formats = ['jpg', 'png', 'bmp', 'tiff', 'webp']
    
    for fmt in formats:
        filename = f"test_save.{fmt}"
        print(f"\n测试保存格式: {fmt}")
        
        try:
            if processor.save_image(filename):
                print(f"  [OK] 保存成功: {filename}")
                # 检查文件是否存在
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    print(f"  - 文件大小: {file_size} 字节")
                else:
                    print(f"  [FAIL] 文件未创建")
            else:
                print(f"  [FAIL] 保存失败")
        except Exception as e:
            print(f"  [ERROR] 保存错误: {e}")


def test_chinese_path():
    """测试中文路径支持"""
    print("\n=== 测试中文路径支持 ===")
    
    processor = BaseImageProcessor()
    
    # 创建中文路径测试图像
    chinese_dir = "测试目录"
    os.makedirs(chinese_dir, exist_ok=True)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processor.processed_image = test_image
    
    chinese_filename = os.path.join(chinese_dir, "测试图像.jpg")
    print(f"测试中文路径: {chinese_filename}")
    
    try:
        if processor.save_image(chinese_filename):
            print(f"  [OK] 中文路径保存成功")
            
            # 测试加载
            if processor.load_image(chinese_filename):
                print(f"  [OK] 中文路径加载成功")
            else:
                print(f"  [FAIL] 中文路径加载失败")
        else:
            print(f"  [FAIL] 中文路径保存失败")
    except Exception as e:
        print(f"  [ERROR] 中文路径错误: {e}")


def cleanup_test_files():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")
    
    # 清理测试图像
    test_files = [
        "test_save.jpg", "test_save.png", "test_save.bmp", 
        "test_save.tiff", "test_save.webp"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"已删除: {filename}")
    
    # 清理测试目录
    import shutil
    test_dirs = ["test_images", "测试目录"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"已删除目录: {test_dir}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("图像格式支持测试")
    print("=" * 60)
    
    try:
        # 运行各项测试
        test_image_loading()
        test_color_conversion()
        test_image_saving()
        test_chinese_path()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    
    finally:
        # 清理测试文件
        cleanup_test_files()


if __name__ == "__main__":
    main()
