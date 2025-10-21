#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强处理工具 - 快速启动脚本
模块化版本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("图像增强处理工具 - 快速启动")
    print("=" * 40)
    print("正在启动模块化版本...")
    
    # 直接导入并运行主程序
    try:
        from main import main as run_main
        run_main()
    except Exception as e:
        print(f"❌ 启动失败：{e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
