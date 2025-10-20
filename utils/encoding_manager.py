#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码管理模块
统一处理UTF-8和ASCII编码格式
"""

import sys
import locale
import codecs
from typing import Optional, Union, TextIO


class EncodingManager:
    """编码管理器"""
    
    def __init__(self):
        self.system_encoding = self._detect_system_encoding()
        self.preferred_encoding = 'utf-8'
        self.fallback_encoding = 'ascii'
        
    def _detect_system_encoding(self) -> str:
        """检测系统编码"""
        try:
            # 尝试获取系统默认编码
            system_encoding = locale.getpreferredencoding()
            if system_encoding:
                return system_encoding.lower()
        except Exception:
            pass
        
        # 回退到常见编码
        if sys.platform.startswith('win'):
            return 'cp1252'  # Windows默认编码
        else:
            return 'utf-8'  # Unix/Linux默认编码
    
    def get_safe_encoding(self, preferred: Optional[str] = None) -> str:
        """
        获取安全的编码格式
        
        Args:
            preferred: 首选编码，如果为None则使用默认
            
        Returns:
            str: 安全的编码格式
        """
        if preferred:
            return preferred.lower()
        
        # 优先使用UTF-8，如果系统不支持则使用ASCII
        if self._is_encoding_supported('utf-8'):
            return 'utf-8'
        elif self._is_encoding_supported('ascii'):
            return 'ascii'
        else:
            return self.system_encoding
    
    def _is_encoding_supported(self, encoding: str) -> bool:
        """检查编码是否被支持"""
        try:
            codecs.lookup(encoding)
            return True
        except LookupError:
            return False
    
    def safe_encode(self, text: str, encoding: Optional[str] = None) -> bytes:
        """
        安全编码字符串
        
        Args:
            text: 要编码的字符串
            encoding: 编码格式，如果为None则自动选择
            
        Returns:
            bytes: 编码后的字节串
        """
        if encoding is None:
            encoding = self.get_safe_encoding()
        
        try:
            return text.encode(encoding)
        except UnicodeEncodeError:
            # 如果编码失败，尝试使用ASCII并忽略错误字符
            try:
                return text.encode('ascii', errors='ignore')
            except UnicodeEncodeError:
                # 最后回退到UTF-8并替换错误字符
                return text.encode('utf-8', errors='replace')
    
    def safe_decode(self, data: bytes, encoding: Optional[str] = None) -> str:
        """
        安全解码字节串
        
        Args:
            data: 要解码的字节串
            encoding: 编码格式，如果为None则自动检测
            
        Returns:
            str: 解码后的字符串
        """
        if encoding is None:
            encoding = self._detect_encoding(data)
        
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            # 如果解码失败，尝试其他编码
            for fallback_encoding in ['utf-8', 'ascii', 'cp1252', 'latin1']:
                try:
                    return data.decode(fallback_encoding)
                except UnicodeDecodeError:
                    continue
            
            # 最后回退到忽略错误
            return data.decode('utf-8', errors='ignore')
    
    def _detect_encoding(self, data: bytes) -> str:
        """检测字节串的编码格式"""
        # 简单的编码检测逻辑
        try:
            # 尝试UTF-8
            data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        try:
            # 尝试ASCII
            data.decode('ascii')
            return 'ascii'
        except UnicodeDecodeError:
            pass
        
        # 回退到系统编码
        return self.system_encoding
    
    def safe_print(self, text: str, encoding: Optional[str] = None, 
                   file: Optional[TextIO] = None) -> None:
        """
        安全打印字符串
        
        Args:
            text: 要打印的字符串
            encoding: 编码格式
            file: 输出文件，默认为stdout
        """
        if file is None:
            file = sys.stdout
        
        try:
            print(text, file=file)
        except UnicodeEncodeError:
            # 如果打印失败，尝试编码后输出
            safe_text = self.safe_encode(text, encoding).decode('utf-8', errors='replace')
            print(safe_text, file=file)
    
    def get_file_encoding(self, filepath: str) -> str:
        """
        检测文件编码
        
        Args:
            filepath: 文件路径
            
        Returns:
            str: 文件编码格式
        """
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(1024)  # 读取前1KB用于检测
                return self._detect_encoding(raw_data)
        except Exception:
            return self.get_safe_encoding()
    
    def read_text_file(self, filepath: str, encoding: Optional[str] = None) -> str:
        """
        安全读取文本文件
        
        Args:
            filepath: 文件路径
            encoding: 编码格式，如果为None则自动检测
            
        Returns:
            str: 文件内容
        """
        if encoding is None:
            encoding = self.get_file_encoding(filepath)
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果指定编码失败，尝试其他编码
            for fallback_encoding in ['utf-8', 'ascii', 'cp1252', 'latin1']:
                try:
                    with open(filepath, 'r', encoding=fallback_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # 最后回退到忽略错误
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    def write_text_file(self, filepath: str, content: str, 
                       encoding: Optional[str] = None) -> bool:
        """
        安全写入文本文件
        
        Args:
            filepath: 文件路径
            content: 文件内容
            encoding: 编码格式，如果为None则使用UTF-8
            
        Returns:
            bool: 写入是否成功
        """
        if encoding is None:
            encoding = self.get_safe_encoding('utf-8')
        
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def get_console_encoding(self) -> str:
        """获取控制台编码"""
        if sys.platform.startswith('win'):
            # Windows控制台编码
            try:
                import locale
                return locale.getpreferredencoding()
            except:
                return 'cp1252'
        else:
            # Unix/Linux控制台编码
            return 'utf-8'
    
    def setup_console_encoding(self) -> None:
        """设置控制台编码"""
        try:
            if sys.platform.startswith('win'):
                # Windows系统设置控制台编码
                import os
                os.system('chcp 65001 > nul 2>&1')  # 设置为UTF-8
        except Exception:
            pass


# 全局编码管理器实例
encoding_manager = EncodingManager()


def safe_print(text: str, encoding: Optional[str] = None, file: Optional[TextIO] = None) -> None:
    """安全打印函数的便捷接口"""
    encoding_manager.safe_print(text, encoding, file)


def safe_encode(text: str, encoding: Optional[str] = None) -> bytes:
    """安全编码函数的便捷接口"""
    return encoding_manager.safe_encode(text, encoding)


def safe_decode(data: bytes, encoding: Optional[str] = None) -> str:
    """安全解码函数的便捷接口"""
    return encoding_manager.safe_decode(data, encoding)


def get_safe_encoding(preferred: Optional[str] = None) -> str:
    """获取安全编码的便捷接口"""
    return encoding_manager.get_safe_encoding(preferred)
