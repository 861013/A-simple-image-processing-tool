#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
管理应用程序的配置参数
支持UTF-8和ASCII编码格式
"""

import json
import os
from typing import Dict, Any

# 导入编码管理器
from utils.encoding_manager import safe_print, encoding_manager


class Config:
    """配置管理类"""
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_default_config()
        self.load_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "app": {
                "name": "图像增强处理工具",
                "version": "2.0.0",
                "author": "AI Assistant"
            },
            "ui": {
                "window_width": 1200,
                "window_height": 800,
                "theme": "default",
                "font_family": "Arial",
                "font_size": 10
            },
            "image": {
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "default_quality": 95
            },
            "processing": {
                "smooth": {
                    "default_method": "gaussian",
                    "default_kernel_size": 5,
                    "max_kernel_size": 15
                },
                "sharpen": {
                    "default_method": "unsharp_mask",
                    "default_strength": 1.0,
                    "max_strength": 3.0
                },
                "emboss": {
                    "default_method": "basic",
                    "default_strength": 1.0,
                    "max_strength": 3.0
                },
                "edge": {
                    "default_method": "canny",
                    "default_low_threshold": 50,
                    "default_high_threshold": 150,
                    "max_threshold": 300
                },
                "chromatic": {
                    "default_method": "basic",
                    "default_intensity": 5,
                    "max_intensity": 20
                }
            },
            "output": {
                "default_directory": "output",
                "filename_prefix": "enhanced",
                "include_timestamp": True,
                "default_format": "jpg"
            },
            "logging": {
                "level": "INFO",
                "file": "app.log",
                "max_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5
            },
            "encoding": {
                "preferred": "utf-8",
                "fallback": "ascii",
                "console_encoding": "utf-8",
                "file_encoding": "utf-8"
            }
        }
    
    def load_config(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                config_content = encoding_manager.read_text_file(self.config_file)
                file_config = json.loads(config_content)
                self.config = self.merge_config(self.config, file_config)
                safe_print(f"配置已从 {self.config_file} 加载")
            except Exception as e:
                safe_print(f"加载配置文件失败: {e}")
                safe_print("使用默认配置")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            config_json = json.dumps(self.config, indent=4, ensure_ascii=False)
            success = encoding_manager.write_text_file(self.config_file, config_json)
            if success:
                safe_print(f"配置已保存到 {self.config_file}")
            return success
        except Exception as e:
            safe_print(f"保存配置文件失败: {e}")
            return False
    
    def merge_config(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default=None):
        """
        获取配置值
        
        Args:
            key_path (str): 配置键路径，如 "ui.window_width"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        
        Args:
            key_path (str): 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_processing_config(self, algorithm: str) -> Dict[str, Any]:
        """
        获取特定算法的处理配置
        
        Args:
            algorithm (str): 算法名称
            
        Returns:
            算法配置字典
        """
        return self.get(f"processing.{algorithm}", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.get("ui", {})
    
    def get_image_config(self) -> Dict[str, Any]:
        """获取图像配置"""
        return self.get("image", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.get("output", {})
    
    def get_encoding_config(self) -> Dict[str, Any]:
        """获取编码配置"""
        return self.get("encoding", {})
    
    def update_encoding_settings(self):
        """更新编码管理器设置"""
        encoding_config = self.get_encoding_config()
        
        if "preferred" in encoding_config:
            encoding_manager.preferred_encoding = encoding_config["preferred"]
        
        if "fallback" in encoding_config:
            encoding_manager.fallback_encoding = encoding_config["fallback"]
    
    def update_processing_config(self, algorithm: str, config: Dict[str, Any]):
        """
        更新特定算法的处理配置
        
        Args:
            algorithm (str): 算法名称
            config (Dict[str, Any]): 新配置
        """
        current_config = self.get_processing_config(algorithm)
        current_config.update(config)
        self.set(f"processing.{algorithm}", current_config)
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.load_default_config()
    
    def export_config(self, export_path: str):
        """
        导出配置到指定路径
        
        Args:
            export_path (str): 导出路径
        """
        try:
            config_json = json.dumps(self.config, indent=4, ensure_ascii=False)
            success = encoding_manager.write_text_file(export_path, config_json)
            if success:
                safe_print(f"配置已导出到 {export_path}")
            return success
        except Exception as e:
            safe_print(f"导出配置失败: {e}")
            return False
    
    def import_config(self, import_path: str):
        """
        从指定路径导入配置
        
        Args:
            import_path (str): 导入路径
        """
        try:
            if not os.path.exists(import_path):
                safe_print(f"配置文件不存在: {import_path}")
                return False
            
            config_content = encoding_manager.read_text_file(import_path)
            imported_config = json.loads(config_content)
            self.config = self.merge_config(self.config, imported_config)
            safe_print(f"配置已从 {import_path} 导入")
            return True
        except Exception as e:
            safe_print(f"导入配置失败: {e}")
            return False


# 全局配置实例
config = Config()
