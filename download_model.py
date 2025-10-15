from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="modelscope", choices=["modelscope", "huggingface"],
                        help="Download source. modelscope or huggingface")
    parser.add_argument('--repo_id', type=str, default="",
                        help="Override repo id if needed. Default uses official.")
    parser.add_argument('--local_dir', type=str, default=None,
                        help="Local directory to store weights. If not set, defaults to <repo>/weights/Logics-Parsing")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")
    
    # 解析本地目录
    if args.local_dir and str(args.local_dir).strip():
        model_dir = os.path.abspath(args.local_dir)
    else:
        model_dir = os.path.join(script_dir, "weights/Logics-Parsing")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    print(f"target local_dir: {model_dir}")

    src = args.type
    if src == "huggingface":
        from huggingface_hub import snapshot_download
        model_name = args.repo_id or "Logics-MLLM/Logics-Parsing"
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    else:
        # 优先尝试 modelscope，失败则回退到 huggingface
        try:
            from modelscope import snapshot_download as ms_snapshot_download
            model_name = args.repo_id or "Alibaba-DT/Logics-Parsing"
            ms_snapshot_download(repo_id=model_name, local_dir=model_dir)
        except Exception as e:
            print(f"[Warn] modelscope download failed ({e}), falling back to huggingface...")
            from huggingface_hub import snapshot_download
            model_name = "Logics-MLLM/Logics-Parsing" if not args.repo_id else args.repo_id
            snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

    print(f"model downloaded to {model_dir}")