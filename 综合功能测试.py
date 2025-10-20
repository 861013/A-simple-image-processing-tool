#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合功能测试脚本
测试所有图像处理功能的完整性和稳定性
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sharpen_filter import SharpenFilter
from modules.emboss_filter import EmbossFilter
from modules.edge_detection import EdgeDetection
from modules.chromatic_aberration import ChromaticAberration
from modules.color_conversion import ColorConversion
from modules.smooth_filter import SmoothFilter
from modules.base_processor import BaseImageProcessor


def create_test_images():
    """创建各种格式的测试图像"""
    test_images = {}
    
    # RGB图像
    test_images['rgb'] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 灰度图像
    test_images['gray'] = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # RGBA图像
    test_images['rgba'] = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    
    # 彩色图像（更真实的测试数据）
    test_images['colorful'] = np.zeros((100, 100, 3), dtype=np.uint8)
    test_images['colorful'][:50, :50] = [255, 0, 0]  # 红色区域
    test_images['colorful'][:50, 50:] = [0, 255, 0]  # 绿色区域
    test_images['colorful'][50:, :50] = [0, 0, 255]  # 蓝色区域
    test_images['colorful'][50:, 50:] = [255, 255, 0]  # 黄色区域
    
    return test_images


def test_image_processing():
    """测试图像处理功能"""
    print("=" * 60)
    print("图像处理功能综合测试")
    print("=" * 60)
    
    test_images = create_test_images()
    results = {}
    
    # 测试锐化滤波
    print("\n1. 测试锐化滤波...")
    sharpen = SharpenFilter()
    sharpen_results = {}
    
    for name, image in test_images.items():
        if len(image.shape) == 2:
            # 灰度图像转换为3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA图像转换为RGB
            image = image[:, :, :3]
        
        sharpen.original_image = image
        sharpen_results[name] = {}
        
        # 测试各种锐化方法
        methods = ['unsharp_mask', 'laplacian', 'high_pass', 'sobel']
        for method in methods:
            try:
                if method == 'unsharp_mask':
                    result = sharpen.process(method, strength=1.0, radius=1.0, threshold=0)
                else:
                    result = sharpen.process(method, strength=1.0)
                sharpen_results[name][method] = result is not None
            except Exception as e:
                sharpen_results[name][method] = False
                print(f"   {name} {method}: 失败 - {e}")
    
    results['sharpen'] = sharpen_results
    print("   锐化滤波测试完成")
    
    # 测试浮雕效果
    print("\n2. 测试浮雕效果...")
    emboss = EmbossFilter()
    emboss_results = {}
    
    for name, image in test_images.items():
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        emboss.original_image = image
        emboss_results[name] = {}
        
        # 测试各种浮雕方法
        methods = ['basic', 'high', 'low', 'relief']
        for method in methods:
            try:
                if method == 'basic':
                    result = emboss.process(method, angle=45)
                elif method == 'relief':
                    result = emboss.process(method, depth=1.0)
                else:
                    result = emboss.process(method, strength=1.0)
                emboss_results[name][method] = result is not None
            except Exception as e:
                emboss_results[name][method] = False
                print(f"   {name} {method}: 失败 - {e}")
    
    results['emboss'] = emboss_results
    print("   浮雕效果测试完成")
    
    # 测试边缘检测
    print("\n3. 测试边缘检测...")
    edge = EdgeDetection()
    edge_results = {}
    
    for name, image in test_images.items():
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        edge.original_image = image
        edge_results[name] = {}
        
        # 测试各种边缘检测方法
        methods = ['canny', 'sobel', 'laplacian', 'prewitt', 'roberts', 'scharr']
        for method in methods:
            try:
                if method == 'canny':
                    result = edge.process(method, low_threshold=50, high_threshold=150, aperture_size=3)
                elif method == 'sobel':
                    result = edge.process(method, dx=1, dy=1, ksize=3)
                elif method == 'laplacian':
                    result = edge.process(method, ksize=3)
                else:
                    result = edge.process(method)
                edge_results[name][method] = result is not None
            except Exception as e:
                edge_results[name][method] = False
                print(f"   {name} {method}: 失败 - {e}")
    
    results['edge'] = edge_results
    print("   边缘检测测试完成")
    
    # 测试色散效果
    print("\n4. 测试色散效果...")
    chromatic = ChromaticAberration()
    chromatic_results = {}
    
    for name, image in test_images.items():
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        chromatic.original_image = image
        chromatic_results[name] = {}
        
        # 测试各种色散方法
        methods = ['basic', 'advanced', 'radial', 'prism']
        for method in methods:
            try:
                if method == 'basic':
                    result = chromatic.process(method, intensity=5, direction='horizontal')
                elif method == 'advanced':
                    result = chromatic.process(method, intensity=5, blur_radius=1)
                elif method == 'radial':
                    result = chromatic.process(method, intensity=5, center=None)
                elif method == 'prism':
                    result = chromatic.process(method, intensity=5, angle=0)
                chromatic_results[name][method] = result is not None
            except Exception as e:
                chromatic_results[name][method] = False
                print(f"   {name} {method}: 失败 - {e}")
    
    results['chromatic'] = chromatic_results
    print("   色散效果测试完成")
    
    # 测试颜色转换
    print("\n5. 测试颜色转换...")
    color_conv = ColorConversion()
    color_results = {}
    
    for name, image in test_images.items():
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        color_conv.original_image = image
        color_results[name] = {}
        
        # 测试各种颜色转换方法
        methods = ['convert_color_space', 'convert_to_grayscale', 'enhance_color']
        for method in methods:
            try:
                if method == 'convert_color_space':
                    result = color_conv.process(method, color_space='HSV')
                elif method == 'convert_to_grayscale':
                    result = color_conv.process(method, gray_method='luminance')
                elif method == 'enhance_color':
                    result = color_conv.process(method, enhancement_factor=1.2)
                color_results[name][method] = result is not None
            except Exception as e:
                color_results[name][method] = False
                print(f"   {name} {method}: 失败 - {e}")
    
    results['color'] = color_results
    print("   颜色转换测试完成")
    
    return results


def test_file_operations():
    """测试文件操作功能"""
    print("\n" + "=" * 60)
    print("文件操作功能测试")
    print("=" * 60)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processor = BaseImageProcessor()
    processor.original_image = test_image
    
    # 测试不同格式的保存和加载
    formats = ['jpg', 'png', 'bmp', 'tiff']
    results = {}
    
    for fmt in formats:
        print(f"\n测试 {fmt.upper()} 格式...")
        filename = f"测试图像.{fmt}"
        
        # 保存
        save_result = processor.save_image(filename)
        print(f"  保存: {'成功' if save_result else '失败'}")
        
        if save_result:
            # 加载
            processor2 = BaseImageProcessor()
            load_result = processor2.load_image(filename)
            print(f"  加载: {'成功' if load_result else '失败'}")
            
            if load_result:
                print(f"  图像尺寸: {processor2.original_image.shape}")
                print(f"  颜色空间: {processor2.color_space}")
            
            # 清理文件
            try:
                os.remove(filename)
                print(f"  清理: 成功")
            except:
                print(f"  清理: 失败")
        
        results[fmt] = {
            'save': save_result,
            'load': load_result if save_result else False
        }
    
    return results


def test_chinese_support():
    """测试中文支持"""
    print("\n" + "=" * 60)
    print("中文支持测试")
    print("=" * 60)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processor = BaseImageProcessor()
    processor.original_image = test_image
    
    # 测试中文文件名
    chinese_files = [
        "测试图像.jpg",
        "图像处理工具.png",
        "中文路径/测试文件.bmp"
    ]
    
    results = {}
    
    for filename in chinese_files:
        print(f"\n测试文件: {filename}")
        
        # 确保目录存在
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        
        # 保存
        save_result = processor.save_image(filename)
        print(f"  保存: {'成功' if save_result else '失败'}")
        
        if save_result:
            # 加载
            processor2 = BaseImageProcessor()
            load_result = processor2.load_image(filename)
            print(f"  加载: {'成功' if load_result else '失败'}")
            
            if load_result:
                print(f"  图像尺寸: {processor2.original_image.shape}")
            
            # 清理文件
            try:
                if os.path.isfile(filename):
                    os.remove(filename)
                if dirname and os.path.exists(dirname):
                    os.rmdir(dirname)
                print(f"  清理: 成功")
            except:
                print(f"  清理: 失败")
        
        results[filename] = {
            'save': save_result,
            'load': load_result if save_result else False
        }
    
    return results


def print_summary(processing_results, file_results, chinese_results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    # 处理功能总结
    print("\n图像处理功能:")
    for category, methods in processing_results.items():
        print(f"  {category}:")
        for image_type, method_results in methods.items():
            success_count = sum(1 for success in method_results.values() if success)
            total_count = len(method_results)
            print(f"    {image_type}: {success_count}/{total_count} 成功")
    
    # 文件操作总结
    print("\n文件操作功能:")
    for fmt, results in file_results.items():
        save_ok = "成功" if results['save'] else "失败"
        load_ok = "成功" if results['load'] else "失败"
        print(f"  {fmt.upper()}: 保存{save_ok} 加载{load_ok}")
    
    # 中文支持总结
    print("\n中文支持:")
    for filename, results in chinese_results.items():
        save_ok = "成功" if results['save'] else "失败"
        load_ok = "成功" if results['load'] else "失败"
        print(f"  {filename}: 保存{save_ok} 加载{load_ok}")


def main():
    """主函数"""
    print("图像增强处理工具 - 综合功能测试")
    print("测试所有功能的完整性和稳定性")
    
    try:
        # 测试图像处理功能
        processing_results = test_image_processing()
        
        # 测试文件操作功能
        file_results = test_file_operations()
        
        # 测试中文支持
        chinese_results = test_chinese_support()
        
        # 打印总结
        print_summary(processing_results, file_results, chinese_results)
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
