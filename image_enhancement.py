#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强处理程序
支持平滑、锐化、浮雕、边缘提取、色散等五种图像增强算法
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel


class ImageEnhancement:
    """图像增强处理类"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
    def load_image(self, image_path):
        """加载图像"""
        try:
            self.image_path = image_path
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError("无法加载图像文件")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            return True
        except Exception as e:
            print(f"加载图像错误: {e}")
            return False
    
    def smooth_filter(self, image, kernel_size=5):
        """平滑滤波（高斯模糊）"""
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def sharpen_filter(self, image, strength=1.0):
        """锐化滤波"""
        # 创建锐化核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        # 应用锐化核
        if len(image.shape) == 3:
            sharpened = cv2.filter2D(image, -1, kernel)
        else:
            sharpened = cv2.filter2D(image, -1, kernel)
        
        # 确保像素值在有效范围内
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def emboss_filter(self, image):
        """浮雕效果"""
        # 浮雕核
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        
        if len(image.shape) == 3:
            # 转换为灰度图进行浮雕处理
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            embossed = cv2.filter2D(gray, -1, kernel)
            embossed = np.clip(embossed, 0, 255).astype(np.uint8)
            # 转换回RGB
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        else:
            embossed = cv2.filter2D(image, -1, kernel)
            return np.clip(embossed, 0, 255).astype(np.uint8)
    
    def edge_detection(self, image, method='canny'):
        """边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if method == 'canny':
            # Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)
        elif method == 'sobel':
            # Sobel边缘检测
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        else:
            # Laplacian边缘检测
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        
        # 转换回RGB
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def chromatic_aberration(self, image, intensity=5):
        """色散效果"""
        if len(image.shape) != 3:
            return image
        
        # 分离RGB通道
        r, g, b = cv2.split(image)
        
        # 对每个通道应用不同的偏移
        rows, cols = r.shape[:2]
        
        # 创建变换矩阵
        M_r = np.float32([[1, 0, intensity], [0, 1, 0]])
        M_g = np.float32([[1, 0, 0], [0, 1, 0]])
        M_b = np.float32([[1, 0, -intensity], [0, 1, 0]])
        
        # 应用变换
        r_shifted = cv2.warpAffine(r, M_r, (cols, rows))
        g_shifted = cv2.warpAffine(g, M_g, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_b, (cols, rows))
        
        # 合并通道
        result = cv2.merge([r_shifted, g_shifted, b_shifted])
        
        return result
    
    def process_image(self, method, **kwargs):
        """处理图像"""
        if self.original_image is None:
            return None
        
        image = self.original_image.copy()
        
        if method == 'smooth':
            kernel_size = kwargs.get('kernel_size', 5)
            self.processed_image = self.smooth_filter(image, kernel_size)
        elif method == 'sharpen':
            strength = kwargs.get('strength', 1.0)
            self.processed_image = self.sharpen_filter(image, strength)
        elif method == 'emboss':
            self.processed_image = self.emboss_filter(image)
        elif method == 'edge':
            edge_method = kwargs.get('edge_method', 'canny')
            self.processed_image = self.edge_detection(image, edge_method)
        elif method == 'chromatic':
            intensity = kwargs.get('intensity', 5)
            self.processed_image = self.chromatic_aberration(image, intensity)
        
        return self.processed_image
    
    def save_image(self, output_path):
        """保存处理后的图像"""
        if self.processed_image is not None:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换颜色空间并保存
            if len(self.processed_image.shape) == 3:
                save_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            else:
                save_image = self.processed_image
            
            cv2.imwrite(output_path, save_image)
            return True
        return False


class ImageEnhancementGUI:
    """图像增强GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强处理工具")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 创建图像处理实例
        self.image_processor = ImageEnhancement()
        
        # 创建输出目录
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像增强处理工具", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 文件操作
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="选择图像", 
                  command=self.load_image).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="保存图像", 
                  command=self.save_image).pack(fill=tk.X)
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 图像处理选项
        process_frame = ttk.LabelFrame(control_frame, text="图像处理", padding="5")
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理算法选择
        self.process_var = tk.StringVar(value="smooth")
        algorithms = [
            ("平滑滤波", "smooth"),
            ("锐化滤波", "sharpen"),
            ("浮雕效果", "emboss"),
            ("边缘检测", "edge"),
            ("色散效果", "chromatic")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(process_frame, text=text, variable=self.process_var, 
                           value=value, command=self.update_parameters).pack(anchor=tk.W)
        
        # 参数控制
        self.params_frame = ttk.Frame(control_frame)
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理按钮
        ttk.Button(control_frame, text="开始处理", 
                  command=self.process_image).pack(fill=tk.X, pady=10)
        
        # 重置按钮
        ttk.Button(control_frame, text="重置图像", 
                  command=self.reset_image).pack(fill=tk.X)
        
        # 图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="5")
        image_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建matplotlib图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title("原始图像")
        self.ax2.set_title("处理后图像")
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # 将matplotlib图形嵌入到tkinter中
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化参数控制
        self.update_parameters()
        
    def update_parameters(self):
        """更新参数控制界面"""
        # 清除现有参数控件
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        method = self.process_var.get()
        
        if method == "smooth":
            ttk.Label(self.params_frame, text="核大小:").pack(anchor=tk.W)
            self.kernel_size_var = tk.IntVar(value=5)
            kernel_scale = ttk.Scale(self.params_frame, from_=3, to=15, 
                                   variable=self.kernel_size_var, orient=tk.HORIZONTAL)
            kernel_scale.pack(fill=tk.X, pady=(0, 5))
            
        elif method == "sharpen":
            ttk.Label(self.params_frame, text="锐化强度:").pack(anchor=tk.W)
            self.strength_var = tk.DoubleVar(value=1.0)
            strength_scale = ttk.Scale(self.params_frame, from_=0.1, to=3.0, 
                                     variable=self.strength_var, orient=tk.HORIZONTAL)
            strength_scale.pack(fill=tk.X, pady=(0, 5))
            
        elif method == "edge":
            ttk.Label(self.params_frame, text="边缘检测方法:").pack(anchor=tk.W)
            self.edge_method_var = tk.StringVar(value="canny")
            edge_combo = ttk.Combobox(self.params_frame, textvariable=self.edge_method_var,
                                    values=["canny", "sobel", "laplacian"], state="readonly")
            edge_combo.pack(fill=tk.X, pady=(0, 5))
            
        elif method == "chromatic":
            ttk.Label(self.params_frame, text="色散强度:").pack(anchor=tk.W)
            self.intensity_var = tk.IntVar(value=5)
            intensity_scale = ttk.Scale(self.params_frame, from_=1, to=20, 
                                      variable=self.intensity_var, orient=tk.HORIZONTAL)
            intensity_scale.pack(fill=tk.X, pady=(0, 5))
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            if self.image_processor.load_image(file_path):
                self.display_images()
                messagebox.showinfo("成功", "图像加载成功！")
            else:
                messagebox.showerror("错误", "无法加载图像文件！")
    
    def process_image(self):
        """处理图像"""
        if self.image_processor.original_image is None:
            messagebox.showwarning("警告", "请先选择图像文件！")
            return
        
        method = self.process_var.get()
        kwargs = {}
        
        if method == "smooth":
            kwargs['kernel_size'] = self.kernel_size_var.get()
        elif method == "sharpen":
            kwargs['strength'] = self.strength_var.get()
        elif method == "edge":
            kwargs['edge_method'] = self.edge_method_var.get()
        elif method == "chromatic":
            kwargs['intensity'] = self.intensity_var.get()
        
        self.image_processor.process_image(method, **kwargs)
        self.display_images()
        messagebox.showinfo("成功", "图像处理完成！")
    
    def display_images(self):
        """显示图像"""
        self.ax1.clear()
        self.ax2.clear()
        
        if self.image_processor.original_image is not None:
            self.ax1.imshow(self.image_processor.original_image)
            self.ax1.set_title("原始图像")
            self.ax1.axis('off')
        
        if self.image_processor.processed_image is not None:
            self.ax2.imshow(self.image_processor.processed_image)
            self.ax2.set_title("处理后图像")
            self.ax2.axis('off')
        
        self.canvas.draw()
    
    def save_image(self):
        """保存图像"""
        if self.image_processor.processed_image is None:
            messagebox.showwarning("警告", "没有处理后的图像可保存！")
            return
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = self.process_var.get()
        filename = f"enhanced_{method}_{timestamp}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        if self.image_processor.save_image(output_path):
            messagebox.showinfo("成功", f"图像已保存到: {output_path}")
        else:
            messagebox.showerror("错误", "保存图像失败！")
    
    def reset_image(self):
        """重置图像"""
        self.image_processor.processed_image = None
        self.display_images()
        messagebox.showinfo("成功", "图像已重置！")


def main():
    """主函数"""
    root = tk.Tk()
    app = ImageEnhancementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
