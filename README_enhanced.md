# 图像增强处理工具 - 多格式支持版本

## 概述

这是一个功能强大的Python图像处理工具，支持多种图像格式和颜色空间的处理与转换。程序采用模块化设计，易于维护和扩展。

## 主要特性

### 🖼️ 支持的图像格式

**输入格式：**
- JPEG (.jpg, .jpeg)
- PNG (.png) - 支持RGBA通道
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)
- GIF (.gif)

**输出格式：**
- JPEG (.jpg, .jpeg) - 可调节质量
- PNG (.png) - 支持透明通道
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp) - 现代压缩格式
- GIF (.gif) - 支持动画

### 🎨 支持的颜色空间

- **RGB** - 标准红绿蓝颜色空间
- **RGBA** - RGB + Alpha透明通道
- **HSV** - 色调、饱和度、明度
- **LAB** - 感知均匀颜色空间
- **YUV** - 亮度色度颜色空间
- **CMYK** - 印刷四色模式
- **灰度** - 单通道灰度图像
- **黑白** - 二值图像
- **调色板** - 索引颜色模式

### 🔧 图像处理功能

1. **平滑滤波**
   - 高斯模糊
   - 均值滤波
   - 中值滤波
   - 双边滤波（保边去噪）

2. **锐化滤波**
   - 反锐化掩模
   - 拉普拉斯算子
   - 高通滤波
   - Sobel算子

3. **特殊效果**
   - 浮雕效果
   - 边缘检测（Canny、Sobel、Laplacian等）
   - 色散效果
   - 颜色增强

4. **颜色空间转换**
   - RGB ↔ HSV
   - RGB ↔ LAB
   - RGB ↔ YUV
   - RGB ↔ 灰度
   - CMYK ↔ RGB

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖库：
- opencv-python >= 4.5.0
- Pillow >= 9.0.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0
- imageio >= 2.19.0 (可选)

## 使用方法

### 1. 图形界面模式

```bash
python main.py
```

启动图形界面，支持：
- 拖拽加载图像文件
- 实时参数调整
- 预览处理效果
- 多种格式保存

### 2. 命令行模式

```bash
python image_enhancement.py
```

### 3. 编程接口

```python
from modules.base_processor import BaseImageProcessor
from modules.color_conversion import ColorConversion

# 创建处理器
processor = BaseImageProcessor()

# 加载图像
processor.load_image("input.jpg")

# 处理图像
processor.process_image("smooth", kernel_size=5)

# 保存结果
processor.save_image("output.png")
```

## 文件结构

```
shuzixinhao2/
├── main.py                 # 主程序入口
├── image_enhancement.py   # 图像增强处理
├── modules/               # 处理模块
│   ├── base_processor.py  # 基础处理器
│   ├── color_conversion.py # 颜色转换
│   ├── smooth_filter.py   # 平滑滤波
│   ├── sharpen_filter.py  # 锐化滤波
│   ├── emboss_filter.py   # 浮雕效果
│   ├── edge_detection.py  # 边缘检测
│   └── chromatic_aberration.py # 色散效果
├── gui/                   # 图形界面
│   ├── main_window.py     # 主窗口
│   ├── control_panel.py   # 控制面板
│   ├── image_display.py   # 图像显示
│   └── image_info_display.py # 图像信息
├── utils/                 # 工具函数
│   ├── image_utils.py     # 图像工具
│   ├── config.py          # 配置管理
│   └── font_config.py     # 字体配置
└── test_image_formats.py  # 格式测试脚本
```

## 技术特点

### 1. 多格式支持
- 使用PIL和OpenCV双重加载机制
- 自动检测图像格式和颜色空间
- 支持中文路径和文件名

### 2. 颜色空间处理
- 智能颜色空间检测
- 多种转换算法
- 保持图像质量

### 3. 模块化设计
- 每个处理功能独立模块
- 易于添加新功能
- 统一的接口设计

### 4. 错误处理
- 完善的异常处理机制
- 友好的错误提示
- 自动降级处理

## 使用示例

### 颜色空间转换

```python
from modules.color_conversion import ColorConversion

converter = ColorConversion()
converter.load_image("input.jpg")

# RGB转HSV
converter.process("convert_color_space", color_space="HSV")
converter.save_image("output_hsv.jpg")

# 增强颜色
converter.process("enhance_color", enhancement_factor=1.5)
converter.save_image("enhanced.jpg")
```

### 图像滤波

```python
from modules.smooth_filter import SmoothFilter

filter_processor = SmoothFilter()
filter_processor.load_image("noisy.jpg")

# 高斯模糊
filter_processor.process("gaussian", kernel_size=7)
filter_processor.save_image("blurred.jpg")

# 双边滤波（保边去噪）
filter_processor.process("bilateral", d=9, sigma_color=75)
filter_processor.save_image("denoised.jpg")
```

## 测试

运行测试脚本验证功能：

```bash
python test_image_formats.py
```

测试内容包括：
- 各种格式的加载和保存
- 颜色空间转换
- 中文路径支持
- 错误处理

## 注意事项

1. **内存使用**：处理大图像时注意内存占用
2. **文件格式**：某些格式转换可能损失质量
3. **中文路径**：确保系统支持UTF-8编码
4. **依赖库**：imageio为可选依赖，用于特殊格式支持

## 更新日志

### v2.0 (当前版本)
- ✅ 支持多种图像格式（JPG、PNG、BMP、TIFF、WebP、GIF）
- ✅ 支持多种颜色空间（RGB、RGBA、HSV、LAB、YUV、CMYK、灰度）
- ✅ 增强的图像加载和保存功能
- ✅ 中文路径和文件名支持
- ✅ 模块化架构设计
- ✅ 完善的错误处理机制

### v1.0
- 基础图像处理功能
- 简单的GUI界面
- 基本的滤波算法

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License
