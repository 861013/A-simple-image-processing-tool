#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块测试脚本
测试所有模块是否能正常导入和工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("正在测试模块导入...")
    
    try:
        # 测试基础模块
        print("1. 测试基础模块...")
        import numpy as np
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt
        import scipy
        import tkinter as tk
        print("   [OK] 基础依赖库导入成功")
        
        # 测试图像处理模块
        print("2. 测试图像处理模块...")
        from modules.base_processor import BaseImageProcessor
        from modules.smooth_filter import SmoothFilter
        from modules.sharpen_filter import SharpenFilter
        from modules.emboss_filter import EmbossFilter
        from modules.edge_detection import EdgeDetection
        from modules.chromatic_aberration import ChromaticAberration
        print("   [OK] 图像处理模块导入成功")
        
        # 将模块保存到全局变量
        globals()['SmoothFilter'] = SmoothFilter
        globals()['SharpenFilter'] = SharpenFilter
        globals()['EmbossFilter'] = EmbossFilter
        globals()['EdgeDetection'] = EdgeDetection
        globals()['ChromaticAberration'] = ChromaticAberration
        
        # 测试GUI模块
        print("3. 测试GUI模块...")
        from gui.image_display import ImageDisplay
        from gui.control_panel import ControlPanel
        from gui.main_window import MainWindow
        print("   [OK] GUI模块导入成功")
        
        # 将GUI模块保存到全局变量
        globals()['ImageDisplay'] = ImageDisplay
        globals()['ControlPanel'] = ControlPanel
        
        # 测试工具模块
        print("4. 测试工具模块...")
        from utils.image_utils import validate_image_path, get_image_info
        from utils.config import config
        print("   [OK] 工具模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] 模块导入失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n正在测试图像处理功能...")
    
    try:
        # 创建测试图像
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 测试平滑滤波
        print("1. 测试平滑滤波...")
        smooth_filter = SmoothFilter()
        smooth_filter.original_image = test_image
        result = smooth_filter.process('gaussian', kernel_size=5)
        if result is not None:
            print("   [OK] 平滑滤波测试成功")
        else:
            print("   [ERROR] 平滑滤波测试失败")
            return False
        
        # 测试锐化滤波
        print("2. 测试锐化滤波...")
        sharpen_filter = SharpenFilter()
        sharpen_filter.original_image = test_image
        result = sharpen_filter.process('unsharp_mask', strength=1.0)
        if result is not None:
            print("   [OK] 锐化滤波测试成功")
        else:
            print("   [ERROR] 锐化滤波测试失败")
            return False
        
        # 测试浮雕效果
        print("3. 测试浮雕效果...")
        emboss_filter = EmbossFilter()
        emboss_filter.original_image = test_image
        result = emboss_filter.process('basic')
        if result is not None:
            print("   [OK] 浮雕效果测试成功")
        else:
            print("   [ERROR] 浮雕效果测试失败")
            return False
        
        # 测试边缘检测
        print("4. 测试边缘检测...")
        edge_filter = EdgeDetection()
        edge_filter.original_image = test_image
        result = edge_filter.process('canny', low_threshold=50, high_threshold=150)
        if result is not None:
            print("   [OK] 边缘检测测试成功")
        else:
            print("   [ERROR] 边缘检测测试失败")
            return False
        
        # 测试色散效果
        print("5. 测试色散效果...")
        chromatic_filter = ChromaticAberration()
        chromatic_filter.original_image = test_image
        result = chromatic_filter.process('basic', intensity=5)
        if result is not None:
            print("   [OK] 色散效果测试成功")
        else:
            print("   [ERROR] 色散效果测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] 图像处理测试失败: {e}")
        return False

def test_gui_components():
    """测试GUI组件"""
    print("\n正在测试GUI组件...")
    
    try:
        import tkinter as tk
        
        # 创建测试窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        # 测试图像显示组件
        print("1. 测试图像显示组件...")
        frame = tk.Frame(root)
        image_display = ImageDisplay(frame)
        print("   [OK] 图像显示组件创建成功")
        
        # 测试控制面板组件
        print("2. 测试控制面板组件...")
        control_panel = ControlPanel(frame)
        print("   [OK] 控制面板组件创建成功")
        
        # 清理
        root.destroy()
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] GUI组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("图像增强处理工具 - 模块测试")
    print("=" * 60)
    
    # 测试模块导入
    if not test_imports():
        print("\n[ERROR] 模块导入测试失败！")
        return False
    
    # 测试图像处理功能
    if not test_image_processing():
        print("\n[ERROR] 图像处理功能测试失败！")
        return False
    
    # 测试GUI组件
    if not test_gui_components():
        print("\n[ERROR] GUI组件测试失败！")
        return False
    
    print("\n" + "=" * 60)
    print("[OK] 所有测试通过！程序可以正常运行。")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
