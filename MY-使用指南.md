# 1-快速入门

> 使用评价：
>
> * 能解析一部分图表，效果不算特别好，有些表格行会缺失。
> * 如果文档中有水印，会被水印干扰。
> * 我本地16G的GPU，推理解析速度有点慢。

## 下载源码

## 1.1-安装：【推荐】

```bash
# 直接在pycharm中创建python3.10的虚拟环境，然后在终端中执行下面两行命令

# 安装pytorch：  GPU（CUDA 12.4）:
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers

pip install -r requirement.txt

```

## 1.2-下载模型权重

```bash

# 【推荐】从huggingface下载模型：
pip install huggingface_hub
python download_model.py -t huggingface --local_dir "E:\llm_models\weights\Logics-Parsing"


# 从Modelscope下载模型：
pip install modelscope
python download_model.py -t modelscope
## 成功后会在weights目录下生成logis_parsing.pth文件
## D:\mySource\cusor-proj\Logics-Parsing\weights\Logics-Parsing


```

## 1.3-推理

```bash
python3 inference.py --image_path PATH_TO_INPUT_IMG --output_path PATH_TO_OUTPUT --model_path PATH_TO_MODEL


# Example
python inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --image_path "D:\mySource\cusor-proj\Logics-Parsing\my_in\page_002.png" --output_path "D:\mySource\cusor-proj\Logics-Parsing\my_out\sample.html" --attn sdpa --dtype auto --max_new_tokens 1024 --max_pixels 786432 

# 如果想用GPU：
.\venv\Scripts\activate

python inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --image_path "D:\mySource\cusor-proj\Logics-Parsing\my_in\page_002.png" --output_path "D:\mySource\cusor-proj\Logics-Parsing\my_out\sample.html" --attn sdpa --dtype auto --device cuda --max_new_tokens 1024 --max_pixels 786432
```

### GPU 优化说明

**当前配置已针对16GB GPU优化：**

- ✅ 显存分配：15GB（模型）+ 1GB（系统预留）
- ✅ 全部使用CUDA加速
- ✅ 自动选择最优数据类型（bf16/fp16）

**如果遇到显存不足或需要其他GPU配置，请参考：**

- 📖 [GPU优化配置说明.md](GPU优化配置说明.md) - 详细的GPU优化指南
- 不同显存大小的配置建议（8GB/12GB/16GB/24GB+）
- 显存监控和故障排查

## 1.4-批量推理

### 功能说明

批量处理脚本 `batch_inference.py` 可以：

1. 按文件名顺序遍历 `my_in` 目录中的所有图片
2. 对每个图片进行解析，生成单独的 HTML 文件
3. 将所有 HTML 文件合并为一个总文件
4. 所有日志详细记录到 `app.log` 文件中

### 使用方法

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 基础用法（使用默认参数）
python batch_inference.py --model_path "E:\llm_models\weights\Logics-Parsing"

# 完整参数示例（推荐配置）
python batch_inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --input_dir "my_in" --output_dir "my_out" --device cuda --dtype auto --attn sdpa --max_new_tokens 1024 --max_pixels 786432

# 自定义输入输出目录
python batch_inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --input_dir "D:\documents\pages" --output_dir "D:\documents\output"
```

### 主要参数说明

**路径参数：**

- `--model_path`: 模型路径（必需）
- `--input_dir`: 输入图片目录（默认: my_in）
- `--output_dir`: 输出HTML目录（默认: my_out）
- `--log_file`: 日志文件路径（默认: app.log）

**推理参数：**

- `--device`: 设备选择 [auto, cuda, cpu]（默认: auto）
- `--dtype`: 数据类型 [auto, bf16, fp16, fp32]（默认: auto）
- `--attn`: 注意力机制 [sdpa, flash_attention_2]（默认: sdpa）
- `--max_new_tokens`: 最大生成token数（默认: 2048）
- `--max_pixels`: 最大像素数（默认: 1024×1024）
- `--min_pixels`: 最小像素数（默认: 3136）

**其他参数：**

- `--no_viz`: 不生成可视化图片
- `--no_merge`: 不合并HTML文件
- `--prompt`: 自定义提示词（默认已优化为忽略水印、页码、页眉页脚）

### 输出文件说明

**单独的HTML文件：**

- `my_out/page_001.html`
- `my_out/page_002.html`
- ...

**合并的HTML文件：**

- `my_out/page_001_至_007.html`（包含所有页面内容）

**可视化图片（如果启用）：**

- `my_out/page_001_vis.png`（带边界框标注）
- `my_out/page_002_vis.png`
- ...

**日志文件：**

- `app.log`（详细记录所有处理步骤、进度和错误信息）

### 日志查看

```bash
# 实时查看日志
Get-Content app.log -Wait -Tail 50

# 查看完整日志
cat app.log

# 搜索错误信息
Select-String -Path app.log -Pattern "错误|ERROR|失败"
```

### 注意事项

1. **文件命名规范**：图片文件建议使用 `page_001.png`、`page_002.png` 格式命名，脚本会自动提取序号
2. **显存要求**：批量处理时模型会一直驻留在显存中，确保GPU有足够显存（建议16GB以上）
3. **处理时间**：每个页面处理时间取决于图片复杂度，通常需要几十秒
4. **中断恢复**：如果处理中断，可以手动删除已处理的文件，重新运行会跳过
5. **日志文件**：`app.log` 会记录每个文件的处理详情，便于追踪问题

### 水印和干扰信息过滤

**默认提示词已优化：**
脚本默认会指示模型**忽略水印、页码、页眉页脚**，只提取正文内容。

**如果仍然提取了水印，可以尝试：**

1. **使用更强的提示词**：

```bash
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --prompt "Extract only the main content from this document. DO NOT include watermarks, page numbers, headers, footers, or any repeated background text. Focus on the actual document content like paragraphs, tables, and images."
```

2. **中文提示词**（可能效果更好）：

```bash
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --prompt "将文档转换为HTML格式。请忽略所有水印、页码、页眉页脚等干扰信息，只提取正文内容，包括文字、表格和图片。"
```

3. **后处理清理**（推荐）：
   使用专门的水印清理脚本 `clean_watermark.py`：

```bash
# 基本用法（清理默认的水印模式）
python clean_watermark.py --input_dir my_out

# 自定义清理规则
python clean_watermark.py --input_dir my_out --pattern "网构码[／/]?\d*" --pattern "<td></td>"

# 不备份原文件（慎用）
python clean_watermark.py --input_dir my_out --no_backup
```

**默认清理的模式包括：**

- `网构码／\d+` - 网构码水印
- `<td>网构码／\d+</td>` - 表格中的网构码
- `第 X 页 共 Y 页` - 中文页码
- `Page X of Y` - 英文页码

**注意：** 原文件会自动备份为 `.bak` 文件，确认无误后可手动删除

### 示例输出

```
2024-10-15 10:30:15 - INFO - 批量文档解析脚本启动
2024-10-15 10:30:15 - INFO - 在目录 my_in 中找到 7 个图片文件
2024-10-15 10:30:20 - INFO - ✓ 模型加载成功
2024-10-15 10:30:21 - INFO - 开始批量处理 7 个文件
2024-10-15 10:30:21 - INFO - 处理进度: [1/7]
2024-10-15 10:30:21 - INFO - 开始处理文件: page_006.png
2024-10-15 10:30:35 - INFO - 解析完成，内容长度: 2543 字符
2024-10-15 10:30:35 - INFO - ✓ 文件 page_006.png 处理成功
...
2024-10-15 10:35:42 - INFO - 批量处理完成
2024-10-15 10:35:42 - INFO - 总文件数: 7
2024-10-15 10:35:42 - INFO - 成功: 7
2024-10-15 10:35:42 - INFO - 失败: 0
2024-10-15 10:35:42 - INFO - 总耗时: 321.45 秒
2024-10-15 10:35:42 - INFO - 平均每个文件: 45.92 秒
2024-10-15 10:35:43 - INFO - ✓ 成功合并所有文件到: my_out/page_006_至_007.html
```
