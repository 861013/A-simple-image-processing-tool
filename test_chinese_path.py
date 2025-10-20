#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文路径图像加载
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_chinese_test_image():
    """创建中文文件名测试图像"""
    print("正在创建中文文件名测试图像...")
    
    # 创建测试图像
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制测试内容
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    draw.ellipse([200, 50, 300, 100], fill='blue', outline='black', width=2)
    
    # 绘制中文文字
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 150), "中文路径测试图像", fill='black', font=font)
    draw.text((50, 180), "Chinese Path Test Image", fill='black', font=font)
    draw.text((50, 210), "支持中文文件名", fill='black', font=font)
    
    # 保存为中文文件名
    chinese_filenames = [
        "测试图像.jpg",
        "图像增强测试.png", 
        "中文文件名测试.jpeg",
        "图像处理工具测试.bmp"
    ]
    
    saved_files = []
    for filename in chinese_filenames:
        try:
            img.save(filename, quality=95)
            saved_files.append(filename)
            print(f"[OK] 已创建: {filename}")
        except Exception as e:
            print(f"[ERROR] 创建失败 {filename}: {e}")
    
    return saved_files

def test_image_loading():
    """测试图像加载功能"""
    print("\n正在测试图像加载功能...")
    
    try:
        from modules.base_processor import BaseImageProcessor
        
        # 创建测试图像
        test_files = create_chinese_test_image()
        
        if not test_files:
            print("[ERROR] 没有创建测试文件")
            return False
        
        # 测试加载中文文件名图像
        processor = BaseImageProcessor()
        
        for filename in test_files:
            print(f"\n测试加载: {filename}")
            if processor.load_image(filename):
                print(f"[OK] 成功加载: {filename}")
                
                # 获取图像信息
                info = processor.get_image_info()
                if info:
                    print(f"  图像尺寸: {info['shape']}")
                    print(f"  图像类型: {info['dtype']}")
                
                # 测试保存
                output_filename = f"处理后的_{filename}"
                if processor.save_image(output_filename):
                    print(f"[OK] 成功保存: {output_filename}")
                else:
                    print(f"[ERROR] 保存失败: {output_filename}")
            else:
                print(f"[ERROR] 加载失败: {filename}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return False

def test_gui_chinese_display():
    """测试GUI中文显示"""
    print("\n正在测试GUI中文显示...")
    
    try:
        import tkinter as tk
        from gui.image_display import ImageDisplay
        from utils.font_config import init_chinese_support
        
        # 初始化中文支持
        init_chinese_support()
        
        # 创建测试窗口
        root = tk.Tk()
        root.title("中文显示测试")
        root.geometry("600x400")
        
        # 创建图像显示组件
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        image_display = ImageDisplay(frame)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        # 显示测试图像
        image_display.display_images(original_image=test_image, processed_image=test_image)
        
        print("[OK] GUI中文显示测试完成")
        print("请检查窗口中的中文标题是否正确显示")
        
        # 3秒后自动关闭
        root.after(3000, root.destroy)
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] GUI中文显示测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("中文路径和字体支持测试")
    print("=" * 60)
    
    # 测试图像加载
    if test_image_loading():
        print("\n[OK] 中文路径图像加载测试通过")
    else:
        print("\n[ERROR] 中文路径图像加载测试失败")
    
    # 测试GUI中文显示
    if test_gui_chinese_display():
        print("\n[OK] GUI中文显示测试通过")
    else:
        print("\n[ERROR] GUI中文显示测试失败")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
