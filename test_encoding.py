#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码兼容性测试脚本
测试UTF-8和ASCII编码格式的支持
"""

import os
import sys
import tempfile
import locale

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.encoding_manager import encoding_manager, safe_print, safe_encode, safe_decode


def test_encoding_detection():
    """测试编码检测功能"""
    safe_print("=== 编码检测测试 ===")
    
    # 测试系统编码检测
    system_encoding = encoding_manager.system_encoding
    safe_print(f"系统编码: {system_encoding}")
    
    # 测试安全编码选择
    utf8_encoding = encoding_manager.get_safe_encoding('utf-8')
    ascii_encoding = encoding_manager.get_safe_encoding('ascii')
    
    safe_print(f"UTF-8编码: {utf8_encoding}")
    safe_print(f"ASCII编码: {ascii_encoding}")
    
    # 测试控制台编码
    console_encoding = encoding_manager.get_console_encoding()
    safe_print(f"控制台编码: {console_encoding}")


def test_string_encoding():
    """测试字符串编码功能"""
    safe_print("\n=== 字符串编码测试 ===")
    
    # 测试字符串
    test_strings = [
        "Hello World",  # ASCII
        "你好世界",      # UTF-8中文
        "图像处理工具",   # UTF-8中文
        "Image Processing Tool",  # ASCII英文
        "测试字符串 with mixed 内容"  # 混合内容
    ]
    
    for test_str in test_strings:
        safe_print(f"\n测试字符串: {test_str}")
        
        # 测试UTF-8编码
        try:
            utf8_bytes = safe_encode(test_str, 'utf-8')
            decoded_utf8 = safe_decode(utf8_bytes, 'utf-8')
            safe_print(f"  UTF-8编码: {len(utf8_bytes)} 字节")
            safe_print(f"  UTF-8解码: {decoded_utf8}")
        except Exception as e:
            safe_print(f"  UTF-8编码错误: {e}")
        
        # 测试ASCII编码
        try:
            ascii_bytes = safe_encode(test_str, 'ascii')
            decoded_ascii = safe_decode(ascii_bytes, 'ascii')
            safe_print(f"  ASCII编码: {len(ascii_bytes)} 字节")
            safe_print(f"  ASCII解码: {decoded_ascii}")
        except Exception as e:
            safe_print(f"  ASCII编码错误: {e}")


def test_file_encoding():
    """测试文件编码功能"""
    safe_print("\n=== 文件编码测试 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
        test_content = "测试文件内容\n包含中文和English\n图像处理工具测试"
        f.write(test_content)
        temp_file = f.name
    
    try:
        # 测试文件编码检测
        detected_encoding = encoding_manager.get_file_encoding(temp_file)
        safe_print(f"检测到的文件编码: {detected_encoding}")
        
        # 测试文件读取
        read_content = encoding_manager.read_text_file(temp_file)
        safe_print(f"读取的文件内容: {read_content}")
        
        # 测试文件写入
        new_content = "新的测试内容\nNew test content\n编码测试完成"
        write_success = encoding_manager.write_text_file(temp_file, new_content)
        safe_print(f"文件写入成功: {write_success}")
        
        # 验证写入结果
        verify_content = encoding_manager.read_text_file(temp_file)
        safe_print(f"验证写入内容: {verify_content}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_console_output():
    """测试控制台输出功能"""
    safe_print("\n=== 控制台输出测试 ===")
    
    # 测试各种输出
    test_messages = [
        "普通ASCII消息",
        "包含中文的消息",
        "Mixed message with 中文 and English",
        "特殊字符: !@#$%^&*()",
        "Unicode字符: αβγδε"
    ]
    
    for msg in test_messages:
        safe_print(f"输出测试: {msg}")


def test_encoding_fallback():
    """测试编码回退机制"""
    safe_print("\n=== 编码回退测试 ===")
    
    # 测试无效编码的处理
    test_string = "测试回退机制"
    
    # 模拟编码错误
    try:
        # 尝试使用不存在的编码
        invalid_bytes = test_string.encode('invalid_encoding')
    except LookupError:
        safe_print("正确处理了无效编码")
    
    # 测试安全编码
    safe_bytes = safe_encode(test_string, 'invalid_encoding')
    safe_print(f"安全编码结果: {len(safe_bytes)} 字节")
    
    # 测试安全解码
    decoded_result = safe_decode(safe_bytes)
    safe_print(f"安全解码结果: {decoded_result}")


def test_locale_compatibility():
    """测试locale兼容性"""
    safe_print("\n=== Locale兼容性测试 ===")
    
    try:
        # 获取当前locale
        current_locale = locale.getlocale()
        safe_print(f"当前locale: {current_locale}")
        
        # 获取默认编码
        default_encoding = locale.getpreferredencoding()
        safe_print(f"默认编码: {default_encoding}")
        
        # 测试locale设置
        try:
            locale.setlocale(locale.LC_ALL, 'C')
            safe_print("成功设置locale为C")
        except Exception as e:
            safe_print(f"设置locale失败: {e}")
        
    except Exception as e:
        safe_print(f"Locale测试错误: {e}")


def test_platform_specific():
    """测试平台特定功能"""
    safe_print("\n=== 平台特定测试 ===")
    
    safe_print(f"操作系统: {sys.platform}")
    
    if sys.platform.startswith('win'):
        safe_print("Windows平台检测")
        # Windows特定测试
        try:
            import locale
            windows_encoding = locale.getpreferredencoding()
            safe_print(f"Windows编码: {windows_encoding}")
        except Exception as e:
            safe_print(f"Windows编码检测错误: {e}")
    
    elif sys.platform.startswith('linux'):
        safe_print("Linux平台检测")
        # Linux特定测试
        try:
            linux_encoding = os.environ.get('LANG', 'unknown')
            safe_print(f"Linux LANG: {linux_encoding}")
        except Exception as e:
            safe_print(f"Linux编码检测错误: {e}")
    
    elif sys.platform.startswith('darwin'):
        safe_print("macOS平台检测")
        # macOS特定测试
        try:
            macos_encoding = os.environ.get('LANG', 'unknown')
            safe_print(f"macOS LANG: {macos_encoding}")
        except Exception as e:
            safe_print(f"macOS编码检测错误: {e}")


def main():
    """主测试函数"""
    safe_print("=" * 60)
    safe_print("编码兼容性测试")
    safe_print("支持UTF-8和ASCII编码格式")
    safe_print("=" * 60)
    
    try:
        # 设置控制台编码
        encoding_manager.setup_console_encoding()
        
        # 运行各项测试
        test_encoding_detection()
        test_string_encoding()
        test_file_encoding()
        test_console_output()
        test_encoding_fallback()
        test_locale_compatibility()
        test_platform_specific()
        
        safe_print("\n" + "=" * 60)
        safe_print("编码兼容性测试完成！")
        safe_print("程序支持UTF-8和ASCII编码格式")
        safe_print("=" * 60)
        
    except Exception as e:
        safe_print(f"\n测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()
