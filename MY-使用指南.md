
# 1-快速入门
## 下载源码


## 1.1-安装：【法二-推荐这个】
``` bash
# 直接在pycharm中创建python3.10的虚拟环境，然后在终端中执行下面两行命令

# 安装pytorch：  GPU（CUDA 12.4）:
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers

pip install -r requirement.txt

```

## 1.2-下载模型权重
``` bash
# 从Modelscope下载模型：
pip install modelscope
python download_model.py -t modelscope
## 成功后会在weights目录下生成logis_parsing.pth文件
## D:\mySource\cusor-proj\Logics-Parsing\weights\Logics-Parsing


# 【推荐】从huggingface下载模型：
pip install huggingface_hub
python download_model.py -t huggingface --local_dir "E:\llm_models\weights\Logics-Parsing"

```

## 1.3-推理
``` bash
python3 inference.py --image_path PATH_TO_INPUT_IMG --output_path PATH_TO_OUTPUT --model_path PATH_TO_MODEL


# Example
python inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --image_path "D:\mySource\cusor-proj\Logics-Parsing\my_in\page_002.png" --output_path "D:\mySource\cusor-proj\Logics-Parsing\my_out\sample.html" --attn sdpa --dtype auto --max_new_tokens 1024 --max_pixels 786432 

# 如果想用GPU，且让程序自动选择设备：
.\venv\Scripts\activate

python inference.py --model_path "E:\llm_models\weights\Logics-Parsing" --image_path "D:\mySource\cusor-proj\Logics-Parsing\my_in\page_002.png" --output_path "D:\mySource\cusor-proj\Logics-Parsing\my_out\sample.html" --attn sdpa --dtype auto --device cuda --max_new_tokens 1024 --max_pixels 786432
```

# 提示词
## 1-修改代码
- 我想把这个项目在本地运行起来，请帮我优化代码，具体情况如下：
  - 项目代码在: D:\mySource\cusor-proj\Logics-Parsing
  - 我是windows环境，gpu只有16G，CUDA是12.4
  - 虚拟环境venv已经创建，python版本是3.10
  - 要先激活虚拟环境，再执行后端的包安装、代码执行等操作。
