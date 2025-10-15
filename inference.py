import torch
import os 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
import math 
import cv2 
import argparse


def inference(img_url, prompt, system_prompt="You are a helpful assistant", *, max_new_tokens=2048, dtype=None, device=None, attn_impl="sdpa", min_pixels=28*28*4, max_pixels=1024*1024):
  # 裁剪/预缩放，降低显存
  image = Image.open(img_url).convert("RGB")
  # 记录原始尺寸用于可视化映射
  orig_w, orig_h = image.size
  # 计算建议尺寸
  new_h, new_w = smart_resize(orig_h, orig_w, min_pixels=min_pixels, max_pixels=max_pixels)
  if (new_w, new_h) != (orig_w, orig_h):
    Resampling = getattr(Image, "Resampling", None)
    resample_method = Resampling.BILINEAR if Resampling is not None else (getattr(Image, "BILINEAR", Image.NEAREST))
    image = image.resize((new_w, new_h), resample=resample_method)

  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # 让 accelerate 处理设备放置，尽量保持在 CPU，模型会在 generate 时分发
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
  
  # 将 inputs 移动到模型所在的设备
  if device is not None and device.startswith("cuda") and torch.cuda.is_available():
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

  gen_kwargs = dict(max_new_tokens=max_new_tokens)

  # 推理期自动混合精度
  amp_dtype = torch.bfloat16 if dtype == torch.bfloat16 else (torch.float16 if dtype == torch.float16 else None)
  with torch.inference_mode():
    if amp_dtype is not None and (device is None or device.startswith("cuda")) and torch.cuda.is_available():
      with torch.autocast("cuda", dtype=amp_dtype):
        output_ids = model.generate(**inputs, **gen_kwargs)
    else:
      output_ids = model.generate(**inputs, **gen_kwargs)

  # 从 inputs 中获取 input_ids（兼容字典和对象两种形式）
  input_ids_tensor = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids_tensor, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

  # Qwen2-VL 的 image_grid_thw 信息用于计算网络视角的宽高，再映射回原图
  image_grid_thw = inputs['image_grid_thw'] if isinstance(inputs, dict) else inputs.image_grid_thw
  input_height = image_grid_thw[0][1]*14
  input_width = image_grid_thw[0][2]*14

  return output_text[0], input_height, input_width



def qwenvl_pred_cast_tag(input_text: str) -> str:
    output = input_text

    IMG_RE = re.compile(
        r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>' ,
        flags=re.IGNORECASE,
    )
    output = IMG_RE.sub('', output)


    output = re.sub(
        r'<p\b[^>]*>(.*?)<\/p>',
        r'\1\n',
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )


    def strip_div(class_name: str, txt: str) -> str:
        pattern = re.compile(
            rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)<\/div>\s*',
            flags=re.DOTALL | re.IGNORECASE,
        )
        return pattern.sub(r' \1 ', txt)

    for cls in ['image', 'chemistry', 'table', 'formula', 'image caption']:
        output = strip_div(cls, output)
    output = output.replace(" </td>", "</td>")
    return output


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 1024*1024
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def plot_bbox(img_path, pred, input_height, input_width, output_path):
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    scale = (img_width / input_width, img_height / input_height)
    bboxes = []

    pattern = re.compile(r'data-bbox="(\d+),(\d+),(\d+),(\d+)"')

    scale_x, scale_y = scale  

    def replace_bbox(match):
        x1, y1, x2, y2 = map(int, match)
        bboxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])

    
    matches = re.findall(pattern, pred)
    if matches:
        for match in matches:
            # print(match)
            replace_bbox(match)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)

    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logics-Parsing for document parsing and visualize the output.")

    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the pre-trained model and processor.")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the input image file for parsing.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the prediction.")
    parser.add_argument("--prompt", type=str, default="QwenVL HTML", 
                        help="The prompt to send to the model. (default: %(default)s)")

    # 新增可配置参数
    parser.add_argument("--attn", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention backend: sdpa (PyTorch) or flash_attention_2 (requires flash-attn).")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Computation dtype for model weights.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device preference. auto will pick cuda if available.")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens to generate. Lower to save memory.")
    parser.add_argument("--min_pixels", type=int, default=28*28*4,
                        help="Minimum pixels after resize to control memory. Must be multiple of 28*28.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024,
                        help="Maximum pixels after resize to control memory.")
    parser.add_argument("--no_viz", action="store_true", help="Disable bbox visualization output.")


    args = parser.parse_args()
    

    model_path = args.model_path
    image_path = args.image_path
    prompt = args.prompt
    output_path =  args.output_path

    # 设备与精度选择
    if args.device == "cuda":
        use_cuda = torch.cuda.is_available()
    elif args.device == "cpu":
        use_cuda = False
    else:  # auto
        use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

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

    attn_impl = args.attn

    # 加载模型（尽量使用 sdpa，若强制 flash 失败则回退）
    try:
        if use_cuda:
            # 使用 CUDA 时，让 device_map="auto" 自动决定最佳分配策略
            # 不设置 max_memory，避免参数被offload到磁盘
            print("使用自动设备映射（device_map=auto）")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map="auto",
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map=None,
            )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"[Warn] flash_attention_2 backend failed ({e}), falling back to sdpa.")
            if use_cuda:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    device_map="auto",
                )
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    device_map=None,
                )
        else:
            raise

    print("model loaded")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    prediction, input_height, input_width = inference(
        image_path,
        prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        device=device,
        attn_impl=attn_impl,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    if not args.no_viz:
        output_img_path = os.path.splitext(image_path)[0] + "_vis.png"
        try:
            plot_bbox(image_path, prediction, input_height, input_width, output_img_path)
        except Exception as e:
            print(f"[Warn] failed to draw bbox visualization: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prediction)

    # 可选后处理
    # prediction = qwenvl_pred_cast_tag(prediction)

    print(prediction)
