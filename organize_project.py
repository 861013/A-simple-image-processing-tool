#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序结构整理脚本
确保代码结构条理得当，支持UTF-8和ASCII编码
"""

import os
import sys
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.encoding_manager import safe_print, encoding_manager


class ProjectOrganizer:
    """项目结构整理器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.encoding_manager = encoding_manager
    
    def check_file_encoding(self, file_path: Path) -> str:
        """检查文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                return self.encoding_manager._detect_encoding(raw_data)
        except Exception:
            return "unknown"
    
    def ensure_utf8_header(self, file_path: Path) -> bool:
        """确保文件有UTF-8编码声明"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已有编码声明
            if '# -*- coding:' in content or '# coding:' in content:
                return True
            
            # 添加编码声明
            lines = content.split('\n')
            if lines[0].startswith('#!/usr/bin/env python3'):
                lines.insert(1, '# -*- coding: utf-8 -*-')
            else:
                lines.insert(0, '# -*- coding: utf-8 -*-')
            
            new_content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
            
        except Exception as e:
            safe_print(f"处理文件 {file_path} 失败: {e}")
            return False
    
    def organize_python_files(self) -> None:
        """整理Python文件"""
        safe_print("=== 整理Python文件 ===")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            safe_print(f"处理文件: {py_file}")
            
            # 检查编码
            encoding = self.check_file_encoding(py_file)
            safe_print(f"  编码格式: {encoding}")
            
            # 确保UTF-8编码声明
            if self.ensure_utf8_header(py_file):
                safe_print(f"  [OK] UTF-8编码声明已添加")
            else:
                safe_print(f"  [FAIL] UTF-8编码声明添加失败")
    
    def organize_project_structure(self) -> None:
        """整理项目结构"""
        safe_print("\n=== 整理项目结构 ===")
        
        # 定义标准目录结构
        standard_dirs = [
            "modules",      # 处理模块
            "gui",          # 图形界面
            "utils",        # 工具函数
            "tests",        # 测试文件
            "docs",         # 文档
            "examples",     # 示例
            "output",       # 输出目录
            "config",       # 配置文件
        ]
        
        for dir_name in standard_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                safe_print(f"创建目录: {dir_name}")
            else:
                safe_print(f"目录已存在: {dir_name}")
    
    def organize_config_files(self) -> None:
        """整理配置文件"""
        safe_print("\n=== 整理配置文件 ===")
        
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # 移动配置文件到config目录
        config_files = [
            "config.json",
            "requirements.txt",
            "README.md",
            "README_enhanced.md",
        ]
        
        for config_file in config_files:
            src_path = self.project_root / config_file
            if src_path.exists():
                dst_path = config_dir / config_file
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    safe_print(f"移动配置文件: {config_file} -> config/")
                else:
                    safe_print(f"配置文件已存在: config/{config_file}")
    
    def organize_test_files(self) -> None:
        """整理测试文件"""
        safe_print("\n=== 整理测试文件 ===")
        
        tests_dir = self.project_root / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # 移动测试文件到tests目录
        test_files = [
            "test_image_formats.py",
            "test_encoding.py",
            "test_chinese_path.py",
            "test_modules.py",
            "demo_features.py",
        ]
        
        for test_file in test_files:
            src_path = self.project_root / test_file
            if src_path.exists():
                dst_path = tests_dir / test_file
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    safe_print(f"移动测试文件: {test_file} -> tests/")
                else:
                    safe_print(f"测试文件已存在: tests/{test_file}")
    
    def organize_documentation(self) -> None:
        """整理文档文件"""
        safe_print("\n=== 整理文档文件 ===")
        
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # 移动文档文件到docs目录
        doc_files = [
            "README.md",
            "README_enhanced.md",
            "完善总结.md",
            "PyCharm运行说明.md",
        ]
        
        for doc_file in doc_files:
            src_path = self.project_root / doc_file
            if src_path.exists():
                dst_path = docs_dir / doc_file
                if not dst_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    safe_print(f"移动文档文件: {doc_file} -> docs/")
                else:
                    safe_print(f"文档文件已存在: docs/{doc_file}")
    
    def create_project_summary(self) -> None:
        """创建项目总结"""
        safe_print("\n=== 创建项目总结 ===")
        
        summary_content = """# 图像处理程序项目总结

## 项目结构

```
shuzixinhao2/
├── main.py                      # 主程序入口
├── image_enhancement.py        # 图像增强处理
├── modules/                    # 处理模块
│   ├── base_processor.py       # 基础处理器
│   ├── color_conversion.py     # 颜色转换
│   ├── smooth_filter.py        # 平滑滤波
│   ├── sharpen_filter.py       # 锐化滤波
│   ├── emboss_filter.py        # 浮雕效果
│   ├── edge_detection.py       # 边缘检测
│   └── chromatic_aberration.py # 色散效果
├── gui/                        # 图形界面
│   ├── main_window.py          # 主窗口
│   ├── control_panel.py        # 控制面板
│   ├── image_display.py        # 图像显示
│   └── image_info_display.py   # 图像信息
├── utils/                      # 工具函数
│   ├── image_utils.py          # 图像工具
│   ├── config.py               # 配置管理
│   ├── font_config.py          # 字体配置
│   └── encoding_manager.py     # 编码管理
├── tests/                      # 测试文件
│   ├── test_image_formats.py   # 格式测试
│   ├── test_encoding.py        # 编码测试
│   └── demo_features.py        # 功能演示
├── docs/                       # 文档
│   ├── README.md               # 说明文档
│   ├── README_enhanced.md      # 增强说明
│   └── 完善总结.md             # 完善总结
├── config/                     # 配置文件
│   ├── config.json             # 程序配置
│   └── requirements.txt        # 依赖列表
└── output/                     # 输出目录
```

## 主要特性

### 编码支持
- ✅ UTF-8编码格式
- ✅ ASCII编码格式
- ✅ 自动编码检测
- ✅ 安全编码转换

### 图像格式支持
- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png) - 支持透明通道
- ✅ BMP (.bmp)
- ✅ TIFF (.tiff, .tif)
- ✅ WebP (.webp)
- ✅ GIF (.gif)

### 颜色空间支持
- ✅ RGB - 标准红绿蓝
- ✅ RGBA - RGB + Alpha透明通道
- ✅ HSV - 色调、饱和度、明度
- ✅ LAB - 感知均匀颜色空间
- ✅ YUV - 亮度色度颜色空间
- ✅ CMYK - 印刷四色模式
- ✅ 灰度 - 单通道灰度图像

### 处理功能
- ✅ 平滑滤波 (高斯、均值、中值、双边)
- ✅ 锐化滤波 (多种核函数)
- ✅ 浮雕效果
- ✅ 边缘检测 (Canny、Sobel、Laplacian)
- ✅ 色散效果
- ✅ 颜色空间转换
- ✅ 图像增强

## 技术特点

1. **模块化设计** - 每个功能独立模块，易于维护
2. **编码兼容** - 支持UTF-8和ASCII编码格式
3. **错误处理** - 完善的异常处理机制
4. **中文支持** - 完全支持中文路径和文件名
5. **跨平台** - 支持Windows、Linux、macOS

## 使用方式

### 图形界面
```bash
python main.py
```

### 命令行
```bash
python image_enhancement.py
```

### 编程接口
```python
from modules.base_processor import BaseImageProcessor
from modules.color_conversion import ColorConversion

processor = BaseImageProcessor()
processor.load_image("input.jpg")
processor.process_image("smooth", kernel_size=5)
processor.save_image("output.png")
```

## 测试验证

运行测试脚本验证功能：
```bash
python tests/test_image_formats.py
python tests/test_encoding.py
python tests/demo_features.py
```

## 更新日志

### v2.1 (当前版本)
- ✅ 支持UTF-8和ASCII编码格式
- ✅ 完善的项目结构整理
- ✅ 统一的编码管理
- ✅ 增强的配置管理
- ✅ 完善的错误处理

### v2.0
- ✅ 支持多种图像格式和颜色空间
- ✅ 模块化架构设计
- ✅ 中文路径支持

### v1.0
- ✅ 基础图像处理功能
- ✅ 简单的GUI界面
"""
        
        summary_file = self.project_root / "PROJECT_SUMMARY.md"
        success = encoding_manager.write_text_file(str(summary_file), summary_content)
        
        if success:
            safe_print(f"项目总结已创建: {summary_file}")
        else:
            safe_print("项目总结创建失败")
    
    def run_organization(self) -> None:
        """运行完整的项目整理"""
        safe_print("=" * 60)
        safe_print("程序结构整理")
        safe_print("支持UTF-8和ASCII编码格式")
        safe_print("=" * 60)
        
        try:
            # 设置控制台编码
            self.encoding_manager.setup_console_encoding()
            
            # 执行整理步骤
            self.organize_project_structure()
            self.organize_python_files()
            self.organize_config_files()
            self.organize_test_files()
            self.organize_documentation()
            self.create_project_summary()
            
            safe_print("\n" + "=" * 60)
            safe_print("程序结构整理完成！")
            safe_print("项目现在支持UTF-8和ASCII编码格式")
            safe_print("代码结构条理得当")
            safe_print("=" * 60)
            
        except Exception as e:
            safe_print(f"\n整理过程中发生错误: {e}")


def main():
    """主函数"""
    organizer = ProjectOrganizer()
    organizer.run_organization()


if __name__ == "__main__":
    main()
