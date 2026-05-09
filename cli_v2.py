import os
import argparse
import sys

# 1. 导入 IndexTTS2 (这会触发原脚本中的缓存路径设置)
from indextts.infer_v2 import IndexTTS2

# 2. 立即覆盖为系统默认路径，防止重复下载
# 这将确保 CLI 使用与 WebUI 相同的 /root/.cache/huggingface 目录
os.environ['HF_HUB_CACHE'] = '/root/.cache/huggingface/hub'
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 解决脚本在不同目录下运行时的模块导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 CLI Inference Tool (V2)")
    
    # Core arguments
    parser.add_argument("-t", "--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to speaker reference audio (.wav)")
    parser.add_argument("-o", "--output", type=str, default="output_cli.wav", help="Output path for generated audio")
    
    # Performance arguments
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference (Highly recommended for RTX GPUs)")
    parser.add_argument("--compile", action="store_true", default=True, help="Enable torch.compile optimization")
    
    # Emotion arguments
    parser.add_argument("--emo", type=str, default=None, help="Path to emotion reference audio (Optional)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Emotion mix alpha (0.0 - 1.0)")
    parser.add_argument("--emo_text", action="store_true", help="Automatically guide emotion based on text content")
    
    # Model configuration
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory containing model weights")
    parser.add_argument("--cfg", type=str, default="checkpoints/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    print(f">> Initializing IndexTTS2 (FP16={args.fp16}, Compile={args.compile})...")
    
    try:
        tts = IndexTTS2(
            cfg_path=args.cfg,
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            use_torch_compile=args.compile
        )

        print(f">> Starting synthesis...")
        print(f"   Text: {args.text}")
        print(f"   Speaker Voice: {args.voice}")
        if args.emo:
            print(f"   Emotion Voice: {args.emo} (Alpha: {args.alpha})")
        if args.emo_text:
            print(f"   Emotion Mode: Auto (Guided by text)")

        # Execute inference
        tts.infer(
            spk_audio_prompt=args.voice,
            text=args.text,
            output_path=args.output,
            emo_audio_prompt=args.emo,
            emo_alpha=args.alpha,
            use_emo_text=args.emo_text,
            verbose=False
        )
        
        print(f"\n>> Success! Audio saved to: {args.output}")

    except Exception as e:
        print(f"\n>> Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
