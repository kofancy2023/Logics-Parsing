# 批量文档解析工具

本工具提供了批量处理文档图片的功能，可以自动遍历目录中的所有图片文件，并生成HTML格式的解析结果。

## 功能特性

✅ **批量处理**：自动遍历目录中的所有图片文件  
✅ **顺序处理**：按文件名从小到大排序处理  
✅ **单文件输出**：每个图片生成独立的HTML文件  
✅ **合并输出**：所有结果自动合并为一个总文件  
✅ **详细日志**：所有处理步骤记录到日志文件  
✅ **可视化**：可选生成带边界框标注的图片  
✅ **错误恢复**：详细的错误信息便于排障  
✅ **进度跟踪**：实时显示处理进度和统计信息  

## 快速开始

### 1. 准备工作

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 确保已安装所有依赖
pip install -r requirement.txt
```

### 2. 准备输入文件

将需要解析的图片文件放入 `my_in` 目录：

```
my_in/
  ├── page_001.png
  ├── page_002.png
  ├── page_003.png
  └── ...
```

**文件命名建议**：使用 `page_001.png`、`page_002.png` 格式，便于自动排序和识别。

### 3. 运行批量处理

```bash
# 基础用法
python batch_inference.py --model_path "E:\llm_models\weights\Logics-Parsing"

# 推荐配置（使用GPU加速）
python batch_inference.py \
  --model_path "E:\llm_models\weights\Logics-Parsing" \
  --input_dir "my_in" \
  --output_dir "my_out" \
  --device cuda \
  --dtype auto \
  --attn sdpa \
  --max_new_tokens 1024 \
  --max_pixels 786432
```

### 4. 查看结果

处理完成后，在输出目录中可以找到：

```
my_out/
  ├── page_001.html          # 单个页面的HTML
  ├── page_002.html
  ├── page_003.html
  ├── page_001_至_003.html   # 合并的总文件
  ├── page_001_vis.png       # 可视化图片（可选）
  ├── page_002_vis.png
  └── page_003_vis.png
```

**日志文件**：`app.log` 记录了详细的处理信息

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--model_path` | 模型文件路径 |

### 路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | my_in | 输入图片目录 |
| `--output_dir` | my_out | 输出HTML目录 |
| `--log_file` | app.log | 日志文件路径 |

### 推理参数

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--device` | auto | auto, cuda, cpu | 运行设备 |
| `--dtype` | auto | auto, bf16, fp16, fp32 | 数据类型 |
| `--attn` | sdpa | sdpa, flash_attention_2 | 注意力机制 |
| `--max_new_tokens` | 2048 | 整数 | 最大生成token数 |
| `--max_pixels` | 1048576 | 整数 | 图片最大像素数 |
| `--min_pixels` | 3136 | 整数 | 图片最小像素数 |

### 其他参数

| 参数 | 说明 |
|------|------|
| `--prompt` | 自定义提示词（默认: "QwenVL HTML"）|
| `--no_viz` | 不生成可视化图片 |
| `--no_merge` | 不合并HTML文件 |

## 日志管理

### 实时查看日志

```bash
# PowerShell
Get-Content app.log -Wait -Tail 50

# Git Bash / WSL
tail -f app.log
```

### 搜索错误信息

```bash
# PowerShell
Select-String -Path app.log -Pattern "错误|ERROR|失败"

# Git Bash / WSL
grep -E "错误|ERROR|失败" app.log
```

### 日志内容示例

```
2024-10-15 10:30:15 - INFO - ================================================================================
2024-10-15 10:30:15 - INFO - 批量文档解析脚本启动
2024-10-15 10:30:15 - INFO - ================================================================================
2024-10-15 10:30:15 - INFO - 输入目录: my_in
2024-10-15 10:30:15 - INFO - 输出目录: my_out
2024-10-15 10:30:15 - INFO - 模型路径: E:\llm_models\weights\Logics-Parsing
2024-10-15 10:30:15 - INFO - 在目录 my_in 中找到 7 个图片文件
2024-10-15 10:30:15 - INFO -   [1] page_006.png
2024-10-15 10:30:15 - INFO -   [2] page_007.png
2024-10-15 10:30:20 - INFO - ✓ 模型加载成功
2024-10-15 10:30:21 - INFO - ✓ 处理器加载成功
2024-10-15 10:30:21 - INFO - ================================================================================
2024-10-15 10:30:21 - INFO - 开始批量处理 7 个文件
2024-10-15 10:30:21 - INFO - ================================================================================
2024-10-15 10:30:21 - INFO - 
处理进度: [1/7]
2024-10-15 10:30:21 - INFO - ================================================================================
2024-10-15 10:30:21 - INFO - 开始处理文件: page_006.png
2024-10-15 10:30:21 - INFO - 文件路径: D:\mySource\cusor-proj\Logics-Parsing\my_in\page_006.png
2024-10-15 10:30:21 - INFO - 输出文件: page_006.html
2024-10-15 10:30:21 - INFO - 开始推理解析...
2024-10-15 10:30:35 - INFO - 解析完成，内容长度: 2543 字符
2024-10-15 10:30:35 - INFO - 内容预览: <div class="page"><div class="text">Chapter 1: Introduction...
2024-10-15 10:30:35 - INFO - 已保存到: D:\mySource\cusor-proj\Logics-Parsing\my_out\page_006.html
2024-10-15 10:30:35 - INFO - ✓ 文件 page_006.png 处理成功
```

## 性能优化建议

### GPU加速

使用CUDA可以显著提升处理速度：

```bash
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --device cuda \
  --dtype bf16  # 或 fp16，减少显存占用
```

### 显存优化

如果显存不足（<16GB），可以尝试：

```bash
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --device cuda \
  --dtype fp16 \
  --max_pixels 524288 \  # 减小图片尺寸
  --max_new_tokens 1024  # 减少生成长度
```

### 批量处理优化

1. **跳过可视化**：使用 `--no_viz` 跳过可视化图片生成，节省时间
2. **单独合并**：使用 `--no_merge` 先完成所有单文件处理，最后手动合并
3. **分批处理**：将大量文件分成多个小批次处理，避免长时间运行

## 故障排除

### 常见问题

#### 1. 显存不足

**错误信息**：`CUDA out of memory`

**解决方案**：
- 降低 `--max_pixels` 参数（如 524288 或 262144）
- 使用 `--dtype fp16` 而非 bf16
- 减小 `--max_new_tokens` 值

#### 2. 文件未找到

**错误信息**：`在目录 xxx 中未找到任何图片文件`

**解决方案**：
- 检查输入目录路径是否正确
- 确认图片文件格式（支持 .png, .jpg, .jpeg）
- 检查文件权限

#### 3. 模型加载失败

**错误信息**：`✗ 模型加载失败`

**解决方案**：
- 检查模型路径是否正确
- 确认模型文件完整（应该有多个 .bin 或 .safetensors 文件）
- 查看详细错误信息：`cat app.log | grep ERROR`

#### 4. 处理中断

如果处理过程中断，可以：
1. 查看 `app.log` 确认已处理的文件
2. 手动删除输出目录中的已生成文件（可选）
3. 重新运行脚本（脚本会重新处理所有文件）

## 高级用法

### 自定义提示词

```bash
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --prompt "请详细解析这个文档，包括表格和公式"
```

### 只处理特定文件

如果只想处理部分文件，可以创建临时目录：

```bash
# 创建临时目录
mkdir temp_input
cp my_in/page_001.png temp_input/
cp my_in/page_002.png temp_input/

# 处理临时目录
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --input_dir temp_input \
  --output_dir temp_output
```

### 自定义合并文件名

合并文件名会自动根据首尾页码生成（如 `page_001_至_010.html`）。如果需要自定义，可以使用 `--no_merge` 跳过自动合并，然后手动合并：

```bash
# 不自动合并
python batch_inference.py \
  --model_path "PATH_TO_MODEL" \
  --no_merge

# 手动合并（使用自定义文件名）
# 可以编写简单的脚本来合并HTML文件
```

## 技术细节

### 处理流程

1. **初始化**：加载模型和处理器到指定设备
2. **文件扫描**：扫描输入目录，获取所有图片文件并排序
3. **逐个处理**：
   - 加载图片
   - 智能调整大小（保持宽高比）
   - 调用模型推理
   - 保存HTML结果
   - 生成可视化图片（可选）
4. **合并输出**：将所有HTML文件合并为一个总文件
5. **统计报告**：输出处理统计信息

### 日志级别

- **INFO**：正常处理信息
- **WARNING**：警告信息（不影响主流程）
- **ERROR**：错误信息（某个文件处理失败）

### 文件格式支持

- **输入**：PNG, JPG, JPEG（大小写不敏感）
- **输出**：HTML, PNG（可视化）

## 贡献与反馈

如有问题或建议，欢迎提交 Issue 或 Pull Request。

## 许可证

本项目遵循与主项目相同的许可证。

