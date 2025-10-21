#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体配置模块
统一处理所有中文字符显示问题
"""

import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import font
import platform
import os


def configure_chinese_fonts():
    """配置中文字体支持"""
    
    # 获取系统信息
    system = platform.system()
    
    # 设置matplotlib中文字体
    if system == "Windows":
        # Windows系统字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        # macOS系统字体
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    matplotlib.rcParams['font.size'] = 10
    
    return chinese_fonts


def get_available_chinese_fonts():
    """获取可用的中文字体列表"""
    try:
        # 获取tkinter可用字体
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        available_fonts = list(font.families())
        root.destroy()
        
        # 过滤中文字体
        chinese_fonts = []
        for font_name in available_fonts:
            if any(keyword in font_name.lower() for keyword in 
                   ['sim', 'microsoft', 'yahei', 'heiti', 'pingfang', 'wenquanyi', 'noto']):
                chinese_fonts.append(font_name)
        
        return chinese_fonts
    except Exception as e:
        print(f"获取字体列表失败: {e}")
        return ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']


def test_chinese_display():
    """测试中文显示效果"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建测试图形
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试中文文本
        test_texts = [
            "原始图像",
            "处理后图像", 
            "图像增强处理工具",
            "平滑滤波",
            "锐化滤波",
            "浮雕效果",
            "边缘检测",
            "色散效果"
        ]
        
        y_positions = np.linspace(0.9, 0.1, len(test_texts))
        
        for i, text in enumerate(test_texts):
            ax.text(0.1, y_positions[i], text, fontsize=12, 
                   transform=ax.transAxes, ha='left')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("中文字体显示测试", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 保存测试图像
        test_path = "chinese_font_test.png"
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 中文字体测试图像已保存: {test_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 中文字体测试失败: {e}")
        return False


def configure_tkinter_chinese():
    """配置tkinter中文字体"""
    try:
        # 获取tkinter根窗口
        root = tk.Tk()
        root.withdraw()
        
        # 获取可用字体
        available_fonts = get_available_chinese_fonts()
        
        if available_fonts:
            # 选择第一个可用的中文字体
            chinese_font = available_fonts[0]
            print(f"[OK] 使用中文字体: {chinese_font}")
        else:
            chinese_font = "TkDefaultFont"
            print("[WARNING] 未找到中文字体，使用默认字体")
        
        # 配置默认字体
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family=chinese_font)
        
        root.destroy()
        return chinese_font
        
    except Exception as e:
        print(f"[ERROR] 配置tkinter中文字体失败: {e}")
        return "TkDefaultFont"


def init_chinese_support():
    """初始化中文支持"""
    print("正在初始化中文支持...")
    
    # 配置matplotlib中文字体
    chinese_fonts = configure_chinese_fonts()
    print(f"[OK] matplotlib中文字体配置完成: {chinese_fonts[0]}")
    
    # 配置tkinter中文字体
    tkinter_font = configure_tkinter_chinese()
    print(f"[OK] tkinter中文字体配置完成: {tkinter_font}")
    
    # 测试中文显示
    if test_chinese_display():
        print("[OK] 中文显示测试通过")
    else:
        print("[WARNING] 中文显示测试失败，但程序仍可正常运行")
    
    print("[OK] 中文支持初始化完成")


if __name__ == "__main__":
    init_chinese_support()
