#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量文档解析脚本
功能：
1. 遍历 my_in 目录下的所有图片文件
2. 依次调用 inference 功能解析每个文件
3. 将每个文件的解析结果保存为单独的 HTML 文件
4. 最后将所有 HTML 文件合并为一个总文件
"""

import torch
import os
import sys
import logging
import re
import glob
from pathlib import Path
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import argparse

# 导入 inference.py 中的函数
from inference import inference, smart_resize, plot_bbox


# 配置日志
def setup_logging(log_file='app.log'):
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_sorted_image_files(input_dir):
    """
    获取输入目录中的所有图片文件，按文件名排序
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        排序后的图片文件路径列表
    """
    logger = logging.getLogger(__name__)
    
    # 支持的图片格式（Windows下大小写不敏感，只用小写避免重复）
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    # 去重（使用set，然后转回list）并按文件名排序
    image_files = sorted(list(set(image_files)))
    
    logger.info(f"在目录 {input_dir} 中找到 {len(image_files)} 个图片文件")
    for i, file in enumerate(image_files, 1):
        logger.info(f"  [{i}] {os.path.basename(file)}")
    
    return image_files


def extract_page_number(filename):
    """
    从文件名中提取页码序号
    
    Args:
        filename: 文件名
        
    Returns:
        页码序号字符串，如 "001", "002"
    """
    # 匹配 page_001.png 这种格式
    match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 如果没有匹配到，尝试提取任何数字序列
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    return None


def process_single_file(image_path, output_dir, model, processor, args, logger):
    """
    处理单个图片文件
    
    Args:
        image_path: 图片文件路径
        output_dir: 输出目录路径
        model: 加载的模型
        processor: 加载的处理器
        args: 命令行参数
        logger: 日志记录器
        
    Returns:
        (是否成功, 输出的HTML文件路径)
    """
    filename = os.path.basename(image_path)
    logger.info(f"=" * 80)
    logger.info(f"开始处理文件: {filename}")
    logger.info(f"文件路径: {image_path}")
    
    try:
        # 提取页码序号
        page_num = extract_page_number(filename)
        if page_num:
            output_filename = f"page_{page_num}.html"
        else:
            # 如果无法提取页码，使用原文件名
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.html"
        
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"输出文件: {output_filename}")
        
        # 调用推理函数
        logger.info("开始推理解析...")
        prediction, input_height, input_width = inference(
            image_path,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype_obj,
            device=args.device_str,
            attn_impl=args.attn,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        
        # 记录解析内容的摘要
        content_preview = prediction[:200] if len(prediction) > 200 else prediction
        logger.info(f"解析完成，内容长度: {len(prediction)} 字符")
        logger.info(f"内容预览: {content_preview}...")
        
        # 保存到HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prediction)
        logger.info(f"已保存到: {output_path}")
        
        # 生成可视化图片（如果需要）
        if not args.no_viz:
            try:
                vis_output_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(output_filename)[0]}_vis.png"
                )
                plot_bbox(image_path, prediction, input_height, input_width, vis_output_path)
                logger.info(f"已生成可视化图片: {vis_output_path}")
            except Exception as e:
                logger.warning(f"生成可视化图片失败: {e}")
        
        logger.info(f"✓ 文件 {filename} 处理成功")
        return True, output_path
        
    except Exception as e:
        logger.error(f"✗ 处理文件 {filename} 时发生错误: {e}", exc_info=True)
        return False, None


def merge_html_files(html_files, output_path, logger):
    """
    将多个HTML文件合并为一个文件
    
    Args:
        html_files: HTML文件路径列表
        output_path: 合并后的输出文件路径
        logger: 日志记录器
    """
    logger.info(f"=" * 80)
    logger.info(f"开始合并 {len(html_files)} 个HTML文件")
    logger.info(f"合并输出文件: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # 写入HTML头部
            outfile.write('<!DOCTYPE html>\n')
            outfile.write('<html lang="zh-CN">\n')
            outfile.write('<head>\n')
            outfile.write('    <meta charset="UTF-8">\n')
            outfile.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
            outfile.write('    <title>文档解析结果汇总</title>\n')
            # outfile.write('    <style>\n')
            # outfile.write('        body { font-family: Arial, sans-serif; margin: 20px; }\n')
            # outfile.write('        .page-separator { border-top: 3px solid #333; margin: 40px 0; padding-top: 20px; }\n')
            # outfile.write('        .page-info { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-radius: 5px; }\n')
            # outfile.write('        .page-content { margin: 20px 0; }\n')
            # outfile.write('    </style>\n')
            
            outfile.write('    <style>\n')
            outfile.write('        body { font-family: Arial, sans-serif; margin: 20px; }\n')
            outfile.write('        .page-separator { border-top: 3px solid #333; margin: 40px 0; padding-top: 20px; }\n')
            outfile.write('        .page-info { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-radius: 5px; }\n')
            outfile.write('        .page-content { margin: 20px 0; }\n')
            outfile.write('        /* 表格样式 - 显示边框 */\n')
            outfile.write('        table { border-collapse: collapse; margin: 10px 0; width: auto; }\n')
            outfile.write('        table, th, td { border: 1px solid #333; }\n')
            outfile.write('        th, td { padding: 8px 12px; text-align: left; }\n')
            outfile.write('        th { background-color: #f5f5f5; font-weight: bold; }\n')
            outfile.write('        tr:nth-child(even) { background-color: #fafafa; }\n')
            outfile.write('        tr:hover { background-color: #f0f0f0; }\n')
            outfile.write('    </style>\n')
            
            outfile.write('</head>\n')
            outfile.write('<body>\n')
            outfile.write(f'    <h1>文档解析结果汇总</h1>\n')
            outfile.write(f'    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            outfile.write(f'    <p>共包含 {len(html_files)} 页内容</p>\n')
            outfile.write('    <hr>\n\n')
            
            # 逐个读取并追加HTML文件内容
            for i, html_file in enumerate(html_files, 1):
                filename = os.path.basename(html_file)
                logger.info(f"  [{i}/{len(html_files)}] 添加文件: {filename}")
                
                outfile.write(f'    <div class="page-separator">\n')
                outfile.write(f'        <div class="page-info">\n')
                outfile.write(f'            <h2>第 {i} 页 - {filename}</h2>\n')
                outfile.write(f'        </div>\n')
                outfile.write(f'        <div class="page-content">\n')
                
                # 读取并写入内容
                try:
                    with open(html_file, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        outfile.write('\n')
                except Exception as e:
                    logger.error(f"读取文件 {filename} 失败: {e}")
                    outfile.write(f'            <p style="color: red;">错误: 无法读取文件内容</p>\n')
                
                outfile.write(f'        </div>\n')
                outfile.write(f'    </div>\n\n')
            
            # 写入HTML尾部
            outfile.write('</body>\n')
            outfile.write('</html>\n')
        
        logger.info(f"✓ 成功合并所有文件到: {output_path}")
        
        # 计算文件大小
        file_size = os.path.getsize(output_path)
        logger.info(f"合并文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
        
    except Exception as e:
        logger.error(f"✗ 合并HTML文件时发生错误: {e}", exc_info=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量处理文档解析")
    
    # 输入输出路径
    parser.add_argument("--input_dir", type=str, default="my_in",
                        help="输入图片目录 (默认: my_in)")
    parser.add_argument("--output_dir", type=str, default="my_out",
                        help="输出HTML目录 (默认: my_out)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    
    # 推理参数
    parser.add_argument("--prompt", type=str, default="Convert this document to HTML. Ignore watermarks, page numbers, and headers/footers. Focus only on the main content.",
                        help="提示词 (默认: 转换文档为HTML，忽略水印)")
    parser.add_argument("--attn", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="注意力机制实现")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="数据类型")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="设备选择")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="最大生成token数")
    parser.add_argument("--min_pixels", type=int, default=28*28*4,
                        help="最小像素数")
    parser.add_argument("--max_pixels", type=int, default=1024*1024,
                        help="最大像素数")
    parser.add_argument("--no_viz", action="store_true",
                        help="不生成可视化图片")
    
    # 批处理参数
    parser.add_argument("--no_merge", action="store_true",
                        help="不合并HTML文件")
    parser.add_argument("--log_file", type=str, default="app.log",
                        help="日志文件路径 (默认: app.log)")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 80)
    logger.info("批量文档解析脚本启动")
    logger.info("=" * 80)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"日志文件: {args.log_file}")
    logger.info(f"设备: {args.device}")
    logger.info(f"数据类型: {args.dtype}")
    logger.info(f"最大tokens: {args.max_new_tokens}")
    logger.info(f"最大像素: {args.max_pixels}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录已准备: {args.output_dir}")
    
    # 获取所有图片文件
    image_files = get_sorted_image_files(args.input_dir)
    
    if not image_files:
        logger.error(f"错误: 在目录 {args.input_dir} 中未找到任何图片文件")
        return
    
    # 设备与精度选择
    if args.device == "cuda":
        use_cuda = torch.cuda.is_available()
    elif args.device == "cpu":
        use_cuda = False
    else:  # auto
        use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    
    logger.info(f"使用设备: {device_str}")
    
    # dtype 决策
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:  # auto
        if use_cuda and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif use_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32
    
    logger.info(f"使用数据类型: {dtype}")
    
    # 保存到args中供后续使用
    args.dtype_obj = dtype
    args.device_str = device_str
    
    # 加载模型
    logger.info("=" * 80)
    logger.info("开始加载模型...")
    
    try:
        if use_cuda:
            # 使用 CUDA 时，让 device_map="auto" 自动决定最佳分配策略
            # 不设置 max_memory，避免参数被offload到磁盘
            logger.info("使用自动设备映射（device_map=auto）")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                attn_implementation=args.attn,
                device_map="auto",
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                attn_implementation=args.attn,
                device_map=None,
            )
        logger.info("✓ 模型加载成功")
    except Exception as e:
        if args.attn == "flash_attention_2":
            logger.warning(f"flash_attention_2 加载失败，回退到 sdpa: {e}")
            if use_cuda:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    device_map="auto",
                )
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    device_map=None,
                )
            logger.info("✓ 模型加载成功 (使用 sdpa)")
        else:
            logger.error(f"✗ 模型加载失败: {e}", exc_info=True)
            return
    
    # 加载处理器
    logger.info("加载处理器...")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    logger.info("✓ 处理器加载成功")
    
    # 将全局变量传递给 inference 模块
    import inference as inf_module
    inf_module.model = model
    inf_module.processor = processor
    
    # 批量处理文件
    logger.info("=" * 80)
    logger.info(f"开始批量处理 {len(image_files)} 个文件")
    logger.info("=" * 80)
    
    processed_files = []
    success_count = 0
    fail_count = 0
    
    start_time = datetime.now()
    
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"\n处理进度: [{i}/{len(image_files)}]")
        success, output_path = process_single_file(
            image_path, args.output_dir, model, processor, args, logger
        )
        
        if success:
            success_count += 1
            processed_files.append(output_path)
        else:
            fail_count += 1
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # 统计信息
    logger.info("=" * 80)
    logger.info("批量处理完成")
    logger.info("=" * 80)
    logger.info(f"总文件数: {len(image_files)}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    if success_count > 0:
        logger.info(f"平均每个文件: {elapsed_time/success_count:.2f} 秒")
    
    # 合并HTML文件
    if not args.no_merge and processed_files:
        # 确定合并文件名
        first_page = extract_page_number(os.path.basename(processed_files[0]))
        last_page = extract_page_number(os.path.basename(processed_files[-1]))
        
        if first_page and last_page:
            merged_filename = f"page_{first_page}_至_{last_page}.html"
        else:
            merged_filename = f"merged_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        merged_output_path = os.path.join(args.output_dir, merged_filename)
        merge_html_files(processed_files, merged_output_path, logger)
    
    logger.info("=" * 80)
    logger.info("所有任务完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

