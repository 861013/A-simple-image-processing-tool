#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖库安装脚本
自动安装项目所需的所有依赖库
"""

import subprocess
import sys
import os

def install_requirements():
    """安装requirements.txt中的依赖库"""
    try:
        print("正在安装依赖库...")
        print("=" * 50)
        
        # 检查requirements.txt是否存在
        if not os.path.exists("requirements.txt"):
            print("错误：找不到requirements.txt文件！")
            return False
        
        # 安装依赖库
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖库安装成功！")
            print("\n安装的库：")
            print("- opencv-python: 图像处理核心库")
            print("- Pillow: Python图像库")
            print("- numpy: 数值计算库")
            print("- matplotlib: 绘图库")
            print("- scipy: 科学计算库")
            print("- tkinter-tooltip: GUI工具提示")
            return True
        else:
            print("❌ 依赖库安装失败！")
            print("错误信息：")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 安装过程中出现错误：{e}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python版本过低！")
        print(f"当前版本：{version.major}.{version.minor}.{version.micro}")
        print("要求版本：Python 3.7 或更高")
        return False
    else:
        print(f"✅ Python版本检查通过：{version.major}.{version.minor}.{version.micro}")
        return True

def main():
    """主函数"""
    print("图像增强处理工具 - 依赖库安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        input("按回车键退出...")
        return
    
    print("\n开始安装依赖库...")
    
    # 安装依赖库
    if install_requirements():
        print("\n" + "=" * 50)
        print("🎉 安装完成！现在可以运行程序了：")
        print("python image_enhancement.py")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ 安装失败！请检查错误信息并重试。")
        print("=" * 50)
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
