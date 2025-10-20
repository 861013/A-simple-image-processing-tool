#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建示例图像用于测试
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image():
    """创建一个示例图像用于测试"""
    # 创建图像尺寸
    width, height = 400, 300
    
    # 创建彩色图像
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制一些几何图形和文字
    # 绘制矩形
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    
    # 绘制圆形
    draw.ellipse([200, 50, 300, 100], fill='blue', outline='black', width=2)
    
    # 绘制三角形
    triangle_points = [(350, 50), (320, 100), (380, 100)]
    draw.polygon(triangle_points, fill='green', outline='black', width=2)
    
    # 绘制线条
    for i in range(0, width, 20):
        draw.line([(i, 120), (i+10, 120)], fill='purple', width=2)
    
    # 绘制文字
    try:
        # 尝试使用默认字体
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 150), "图像增强测试", fill='black', font=font)
    draw.text((50, 180), "Image Enhancement Test", fill='black', font=font)
    
    # 绘制一些噪声点
    for i in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(200, height)
        color = tuple(np.random.randint(0, 256, 3))
        draw.point((x, y), fill=color)
    
    # 保存图像
    sample_path = "sample_image.jpg"
    img.save(sample_path, "JPEG", quality=95)
    
    print(f"[OK] 示例图像已创建：{sample_path}")
    print("您可以使用此图像来测试图像增强功能")
    
    return sample_path

if __name__ == "__main__":
    create_sample_image()
