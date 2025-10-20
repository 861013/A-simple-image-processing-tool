#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强处理工具 - 主程序入口
模块化版本，便于维护和扩展
支持UTF-8和ASCII编码格式
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入编码管理器
from utils.encoding_manager import encoding_manager, safe_print

# 设置控制台编码
encoding_manager.setup_console_encoding()

from gui.main_window import MainWindow
from utils.config import config
from utils.font_config import init_chinese_support


def check_dependencies():
    """检查依赖库是否已安装"""
    required_packages = [
        'cv2', 'PIL', 'numpy', 'matplotlib', 'scipy', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2  # type: ignore
            elif package == 'PIL':
                from PIL import Image  # type: ignore
            elif package == 'tkinter':
                import tkinter  # type: ignore
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages


def show_dependency_error(missing_packages):
    """显示依赖库错误信息"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    error_msg = "缺少以下依赖库：\n\n"
    for package in missing_packages:
        error_msg += f"  • {package}\n"
    
    error_msg += "\n请先运行安装脚本：\n"
    error_msg += "python install_dependencies.py\n\n"
    error_msg += "或者手动安装：\n"
    error_msg += "pip install -r requirements.txt"
    
    messagebox.showerror("依赖库缺失", error_msg)
    root.destroy()


def create_sample_image():
    """创建示例图像"""
    try:
        from create_sample_image import create_sample_image
        sample_path = create_sample_image()
        print(f"[OK] 示例图像已创建：{sample_path}")
        return sample_path
    except Exception as e:
        print(f"[ERROR] 创建示例图像失败：{e}")
        return None


def main():
    """主函数"""
    safe_print("=" * 60)
    safe_print("图像增强处理工具 - 模块化版本")
    safe_print("支持UTF-8和ASCII编码格式")
    safe_print("=" * 60)
    
    # 检查依赖库
    safe_print("正在检查依赖库...")
    missing = check_dependencies()
    
    if missing:
        safe_print("[ERROR] 缺少以下依赖库：")
        for package in missing:
            safe_print(f"  - {package}")
        safe_print("\n请先运行安装脚本：")
        safe_print("python install_dependencies.py")
        safe_print("\n或者手动安装：")
        safe_print("pip install -r requirements.txt")
        
        # 显示GUI错误对话框
        show_dependency_error(missing)
        return
    
    safe_print("[OK] 依赖库检查通过！")
    
    # 初始化中文支持
    init_chinese_support()
    
    # 创建输出目录
    output_dir = config.get("output.default_directory", "output")
    os.makedirs(output_dir, exist_ok=True)
    safe_print(f"[OK] 输出目录已创建：{output_dir}")
    
    # 创建示例图像（可选）
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_image()
        return
    
    # 启动主程序
    safe_print("正在启动图像增强处理工具...")
    safe_print("=" * 60)
    
    try:
        # 创建主窗口
        root = tk.Tk()
        app = MainWindow(root)
        
        # 设置窗口图标（如果有的话）
        try:
            root.iconbitmap("icon.ico")
        except:
            pass  # 忽略图标加载失败
        
        # 运行主循环
        app.run()
        
    except Exception as e:
        safe_print(f"[ERROR] 启动失败：{e}")
        messagebox.showerror("启动错误", f"程序启动失败：{str(e)}")
        return
    
    safe_print("程序已退出")


if __name__ == "__main__":
    main()
