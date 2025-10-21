# PyCharm 运行说明

## 问题解决

### 1. 依赖库安装
如果遇到 "No module named 'matplotlib'" 等错误，请先安装依赖库：

**方法一：使用终端**
```bash
pip install -r requirements.txt
```

**方法二：使用PyCharm包管理器**
1. 打开 File -> Settings
2. 选择 Project -> Python Interpreter
3. 点击 "+" 号添加包
4. 搜索并安装以下包：
   - opencv-python
   - Pillow
   - numpy
   - matplotlib
   - scipy

### 2. 运行程序

**推荐方式：使用 start.py**
```bash
python start.py
```

**或者直接运行主程序：**
```bash
python main.py
```

**创建示例图像：**
```bash
python main.py --create-sample
```

### 3. 编码问题解决

如果遇到中文字符显示问题，请在PyCharm中设置：

1. 打开 File -> Settings
2. 选择 Editor -> File Encodings
3. 设置以下编码：
   - Global Encoding: UTF-8
   - Project Encoding: UTF-8
   - Default encoding for properties files: UTF-8
4. 勾选 "Transparent native-to-ascii conversion"

### 4. 模块导入问题

如果遇到模块导入错误，请确保：

1. 项目根目录已设置为源代码根目录
2. 在PyCharm中右键点击项目根目录
3. 选择 "Mark Directory as" -> "Sources Root"

### 5. 测试程序功能

运行测试脚本验证所有模块：
```bash
python test_modules.py
```

## 项目结构

```
shuzixinhao2/
├── main.py                    # 主程序入口
├── start.py                   # 简化启动脚本（推荐）
├── run.py                     # 快速启动脚本
├── test_modules.py            # 模块测试脚本
├── requirements.txt           # 依赖库列表
├── modules/                   # 图像处理算法模块
├── gui/                       # GUI界面模块
└── utils/                     # 工具函数模块
```

## 功能特性

- ✅ 五种图像增强算法（平滑、锐化、浮雕、边缘检测、色散）
- ✅ 每种算法支持多种实现方法
- ✅ 模块化设计，便于维护和扩展
- ✅ 直观的图形用户界面
- ✅ 实时参数调节和对比显示
- ✅ 一键保存处理结果

## 使用说明

1. 运行 `python start.py` 启动程序
2. 点击"选择图像"加载要处理的图片
3. 选择处理算法并调整参数
4. 点击"开始处理"查看效果
5. 点击"保存图像"保存结果到output文件夹

## 故障排除

### 常见问题

1. **模块导入错误**
   - 确保所有依赖库已安装
   - 检查Python解释器设置
   - 重新启动PyCharm

2. **编码问题**
   - 设置文件编码为UTF-8
   - 检查控制台编码设置

3. **GUI显示问题**
   - 确保matplotlib正确安装
   - 检查显示设置

4. **图像处理错误**
   - 确保opencv-python正确安装
   - 检查图像文件格式

如有其他问题，请检查控制台输出的详细错误信息。
