#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水印清理脚本
用于清理已生成的HTML文件中的水印文字
"""

import os
import re
import argparse
from pathlib import Path


def clean_watermark_from_file(file_path, patterns, backup=True):
    """
    从单个HTML文件中清理水印
    
    Args:
        file_path: HTML文件路径
        patterns: 要清理的水印正则表达式列表
        backup: 是否备份原文件
    
    Returns:
        (是否修改, 清理的匹配数)
    """
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        total_matches = 0
        
        # 应用所有清理规则
        for pattern in patterns:
            matches = len(re.findall(pattern, content))
            if matches > 0:
                print(f"  在 {os.path.basename(file_path)} 中找到 {matches} 处匹配: {pattern}")
                content = re.sub(pattern, '', content)
                total_matches += matches
        
        # 如果有修改
        if content != original_content:
            # 备份原文件
            if backup:
                backup_path = file_path + '.bak'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"  ✓ 已备份原文件到: {backup_path}")
            
            # 写入清理后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, total_matches
        else:
            return False, 0
            
    except Exception as e:
        print(f"  ✗ 处理文件失败: {e}")
        return False, 0


def clean_watermarks(input_dir, patterns, file_pattern='*.html', backup=True):
    """
    批量清理目录中所有HTML文件的水印
    
    Args:
        input_dir: 输入目录
        patterns: 水印正则表达式列表
        file_pattern: 文件匹配模式
        backup: 是否备份原文件
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 目录 {input_dir} 不存在")
        return
    
    # 查找所有HTML文件
    html_files = list(input_path.glob(file_pattern))
    
    if not html_files:
        print(f"在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return
    
    print(f"找到 {len(html_files)} 个HTML文件")
    print("=" * 80)
    
    modified_count = 0
    total_matches = 0
    
    for html_file in html_files:
        print(f"\n处理: {html_file.name}")
        modified, matches = clean_watermark_from_file(
            str(html_file), patterns, backup
        )
        
        if modified:
            modified_count += 1
            total_matches += matches
            print(f"  ✓ 已清理 {matches} 处水印")
        else:
            print(f"  - 未发现水印")
    
    print("\n" + "=" * 80)
    print(f"清理完成!")
    print(f"处理文件数: {len(html_files)}")
    print(f"修改文件数: {modified_count}")
    print(f"清理匹配数: {total_matches}")
    
    if backup and modified_count > 0:
        print(f"\n提示: 原文件已备份为 .bak 文件，确认无误后可手动删除")


def main():
    parser = argparse.ArgumentParser(
        description="清理HTML文件中的水印和干扰信息"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="my_out",
        help="输入HTML文件目录（默认: my_out）"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        action='append',
        help="要清理的水印正则表达式（可多次使用添加多个模式）"
    )
    
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.html",
        help="文件匹配模式（默认: *.html）"
    )
    
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份原文件（慎用）"
    )
    
    args = parser.parse_args()
    
    # 默认的水印模式
    default_patterns = [
        r'网构码／\d+',  # 网构码水印
        r'<td>网构码／\d+</td>',  # 表格中的网构码
        r'第\s*\d+\s*页\s*共\s*\d+\s*页',  # 页码
        r'Page\s+\d+\s+of\s+\d+',  # 英文页码
    ]
    
    # 如果用户提供了自定义模式，使用用户模式；否则使用默认模式
    patterns = args.pattern if args.pattern else default_patterns
    
    print("=" * 80)
    print("HTML 水印清理工具")
    print("=" * 80)
    print(f"输入目录: {args.input_dir}")
    print(f"文件模式: {args.file_pattern}")
    print(f"备份原文件: {'否' if args.no_backup else '是'}")
    print(f"\n清理规则:")
    for i, pattern in enumerate(patterns, 1):
        print(f"  {i}. {pattern}")
    print("=" * 80)
    
    clean_watermarks(
        args.input_dir,
        patterns,
        args.file_pattern,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()

