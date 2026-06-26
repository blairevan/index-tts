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


def format_seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis -= 1000
        secs += 1
    if secs >= 60:
        secs -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(timestamps):
    srt_lines = []
    for i, item in enumerate(timestamps, 1):
        start_str = format_seconds_to_srt_time(item["start"])
        end_str = format_seconds_to_srt_time(item["end"])
        text = item["text"]
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)



def split_by_max_length(text, max_len):
    if len(text) <= max_len:
        return [text]
        
    try:
        import jieba
        # Disable jieba default verbose output
        jieba.setLogLevel(20)
        words = []
        raw_tokens = jieba.lcut(text)
        # Merge sequential English/numeric/dot/hyphen tokens to protect filenames, decimals, etc.
        temp_eng = ""
        for token in raw_tokens:
            if token.strip() and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for c in token):
                temp_eng += token
            else:
                if temp_eng:
                    words.append(temp_eng)
                    temp_eng = ""
                words.append(token)
        if temp_eng:
            words.append(temp_eng)
    except ImportError:
        # Fallback to simple split if jieba is not available
        if " " in text:
            words = text.split(" ")
            new_words = []
            for i, w in enumerate(words):
                if i > 0:
                    new_words.append(" ")
                new_words.append(w)
            words = new_words
        else:
            words = list(text)

    sub_texts = []
    current_line = ""
    for word in words:
        if not word:
            continue
        if len(current_line) + len(word) <= max_len:
            current_line += word
        else:
            if current_line:
                sub_texts.append(current_line)
            if len(word) > max_len:
                # If a single word itself is longer than max_len, hard-split it by character
                for i in range(0, len(word), max_len):
                    sub_texts.append(word[i : i + max_len])
                current_line = ""
            else:
                current_line = word
    if current_line:
        sub_texts.append(current_line)
        
    # Strip spaces from start/end of each line for clean rendering
    return [line.strip() for line in sub_texts if line.strip()]


def split_subtitle_item(item, max_char_len=80):
    import re
    text = item["text"]
    start = item["start"]
    end = item["end"]
    duration = end - start
    
    # Split by commas, periods, or other marks
    # We only split on English periods '.' and colons ':' if they are followed by whitespace or end of string
    # (to avoid splitting file names like gpt.pth or decimal numbers like 1.5)
    parts = re.split(r"([,\uff0c\u3002\uff1a\uff1b!?\n\uff01\uff1f;]|\.(?:\s|$)|:(?:\s|$))", text)
    
    sub_texts = []
    current_part = ""
    for part in parts:
        if not part:
            continue
        # Check if the matched part is a punctuation mark
        is_punc = (part in [",", "\uff0c", "\u3002", "\uff1a", "\uff1b", "!", "?", "\n", "\uff01", "\uff1f", ";"] or 
                   part.startswith(".") or 
                   part.startswith(":"))
        if is_punc:
            current_part += part
            sub_texts.append(current_part.strip())
            current_part = ""
        else:
            if current_part:
                sub_texts.append(current_part.strip())
            current_part = part
    if current_part:
        sub_texts.append(current_part.strip())
        
    sub_texts = [t for t in sub_texts if t]
    
    # Apply maximum character length restriction to each segment
    final_sub_texts = []
    for t in sub_texts:
        if len(t) > max_char_len:
            final_sub_texts.extend(split_by_max_length(t, max_char_len))
        else:
            final_sub_texts.append(t)
            
    if len(final_sub_texts) <= 1:
        # If it was split into only 1 item, return the original item with possibly stripped text
        if final_sub_texts:
            item["text"] = final_sub_texts[0]
        return [item]
        
    lengths = [len(t) for t in final_sub_texts]
    total_length = sum(lengths)
    
    if total_length == 0:
        return [item]
        
    sub_items = []
    accumulated_time = start
    for t, l in zip(final_sub_texts, lengths):
        part_duration = duration * (l / total_length)
        sub_items.append({
            "start": round(accumulated_time, 3),
            "end": round(accumulated_time + part_duration, 3),
            "text": t
        })
        accumulated_time += part_duration
        
    return sub_items


def split_all_timestamps(timestamps):
    new_timestamps = []
    for item in timestamps:
        new_timestamps.extend(split_subtitle_item(item))
    return new_timestamps


def parse_text_actions(text):
    import re
    # Match pause pattern: [pause:X.X]
    pattern = r"\[pause:(\d+(?:\.\d+)?)\]"
    parts = re.split(pattern, text)
    
    actions = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                actions.append(('text', part))
        else:
            actions.append(('pause', float(part)))
    return actions


def clean_text_for_synthesis(text):
    import re
    # Convert [replace:原文|替换词] -> 替换词
    text = re.sub(r'\[replace:([^|\]]+)\|([^\]]+)\]', lambda m: f" {m.group(2)} ", text)
    # Convert [汉字|拼音] -> 拼音
    text = re.sub(r'\[([^|\]]+)\|([^\]]+)\]', lambda m: f" {m.group(2)} ", text)
    # Convert [connect:文字] -> 文字
    text = re.sub(r'\[connect:(.*?)\]', r'\1', text)
    return text


def clean_text_for_subtitles(text):
    import re
    # Convert [replace:原文|替换词] -> 原文
    text = re.sub(r'\[replace:([^|\]]+)\|([^\]]+)\]', r'\1', text)
    # Convert [汉字|拼音] -> 汉字
    text = re.sub(r'\[([^|\]]+)\|([^\]]+)\]', r'\1', text)
    # Convert [connect:文字] -> 文字
    text = re.sub(r'\[connect:(.*?)\]', r'\1', text)
    return text


def concatenate_audio_actions(actions, tts, voice, emo, alpha, emo_text, max_tokens, speed, output_path, need_timestamps, max_char_len):
    import torch
    import torchaudio
    import tempfile
    
    waveforms = []
    sample_rate = 22050
    current_time = 0.0
    all_timestamps = []
    
    # Create temporary directory for segment synthesis
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, action in enumerate(actions):
            action_type, val = action
            if action_type == 'text':
                synth_text = clean_text_for_synthesis(val)
                sub_text = clean_text_for_subtitles(val)
                
                temp_segment_path = os.path.join(temp_dir, f"seg_{idx}.wav")
                
                # Execute segment inference
                tts.infer(
                    spk_audio_prompt=voice,
                    text=synth_text,
                    output_path=temp_segment_path,
                    emo_audio_prompt=emo,
                    emo_alpha=alpha,
                    use_emo_text=emo_text,
                    max_text_tokens_per_segment=max_tokens,
                    verbose=False,
                    return_timestamps=need_timestamps
                )
                
                waveform, sr = torchaudio.load(temp_segment_path)
                sample_rate = sr
                duration = waveform.shape[-1] / sr
                
                if need_timestamps:
                    # Segment subtitle split over its exact duration
                    seg_subtitles = split_subtitle_item({
                        "start": 0.0,
                        "end": duration,
                        "text": sub_text
                    }, max_char_len=max_char_len)
                    
                    # Shift segment subtitles by current_time
                    for sub in seg_subtitles:
                        sub["start"] = round(current_time + sub["start"], 3)
                        sub["end"] = round(current_time + sub["end"], 3)
                        all_timestamps.append(sub)
                
                waveforms.append(waveform)
                current_time += duration
                
            elif action_type == 'pause':
                pause_duration = val
                num_channels = waveforms[-1].shape[0] if waveforms else 1
                silence_samples = int(sample_rate * pause_duration)
                silence_tensor = torch.zeros(num_channels, silence_samples, dtype=torch.float32)
                waveforms.append(silence_tensor)
                current_time += pause_duration
                
    if waveforms:
        final_waveform = torch.cat(waveforms, dim=-1)
        if os.path.dirname(output_path) != "":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, final_waveform, sample_rate)
        
        if speed != 1.0:
            change_audio_speed(output_path, speed)
            if need_timestamps:
                for sub in all_timestamps:
                    sub["start"] = round(sub["start"] / speed, 3)
                    sub["end"] = round(sub["end"] / speed, 3)
                    
    return all_timestamps



def change_audio_speed(audio_path, speed):
    if speed == 1.0:
        return
    import torchaudio
    import librosa
    import torch
    
    wav, sr = torchaudio.load(audio_path)
    stretched_channels = []
    for c in range(wav.shape[0]):
        y = wav[c].numpy()
        y_stretched = librosa.effects.time_stretch(y, rate=speed)
        stretched_channels.append(torch.tensor(y_stretched, dtype=torch.float32))
    
    wav_stretched = torch.stack(stretched_channels, dim=0)
    torchaudio.save(audio_path, wav_stretched, sr)


def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 CLI Inference Tool (V2)")
    
    # Core arguments
    parser.add_argument("-t", "--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to speaker reference audio (.wav)")
    parser.add_argument("-o", "--output", type=str, default="output_cli.wav", help="Output path for generated audio")
    parser.add_argument("--speed", type=float, default=1.0, help="Audio playback speed factor (e.g. 1.2 or 0.8)")
    
    # Performance arguments
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference (Highly recommended for RTX GPUs)")
    parser.add_argument("--compile", action="store_true", default=True, help="Enable torch.compile optimization")
    
    # Emotion arguments
    parser.add_argument("--emo", type=str, default=None, help="Path to emotion reference audio (Optional)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Emotion mix alpha (0.0 - 1.0)")
    parser.add_argument("--emo_text", action="store_true", help="Automatically guide emotion based on text content")
    
    # Subtitle arguments
    parser.add_argument("--srt", type=str, default=None, help="Output path for SRT subtitle file (Optional)")
    parser.add_argument("--json_subtitle", type=str, default=None, help="Output path for JSON subtitle file (Optional)")
    parser.add_argument("--max_tokens", type=int, default=120, help="Max tokens per segment (lower values like 20-30 force shorter subtitle segments)")
    parser.add_argument("--max_char_len", type=int, default=80, help="Max character length per subtitle line (lower values like 40 force split on long lines)")

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

        need_timestamps = bool(args.srt or args.json_subtitle)

        # Parse text actions for pauses and pronunciation markup
        actions = parse_text_actions(args.text)
        print(f">> Parsed actions: {actions}")

        # Execute inference and audio concatenation
        timestamps = concatenate_audio_actions(
            actions=actions,
            tts=tts,
            voice=args.voice,
            emo=args.emo,
            alpha=args.alpha,
            emo_text=args.emo_text,
            max_tokens=args.max_tokens,
            speed=args.speed,
            output_path=args.output,
            need_timestamps=need_timestamps,
            max_char_len=args.max_char_len
        )

        print(f"\n>> Success! Audio saved to: {args.output}")

        if need_timestamps and timestamps:
            print("\n>> Generated Subtitles:")
            for ts in timestamps:
                print(f"   [{ts['start']:.3f}s -> {ts['end']:.3f}s] {ts['text']}")

            if args.srt:
                srt_content = generate_srt(timestamps)
                # Ensure parent directories exist
                if os.path.dirname(args.srt) != "":
                    os.makedirs(os.path.dirname(args.srt), exist_ok=True)
                with open(args.srt, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print(f">> SRT subtitle saved to: {args.srt}")

            if args.json_subtitle:
                import json
                
                # Fetch audio metadata using torchaudio, fallback to estimates if it fails
                try:
                    import torchaudio
                    info = torchaudio.info(args.output)
                    sample_rate = info.sample_rate
                    total_seconds = round(info.num_frames / info.sample_rate, 2)
                except Exception:
                    sample_rate = 22050
                    total_seconds = round(timestamps[-1]["end"], 2) if timestamps else 0.0

                json_data = {
                    "audio_file": os.path.basename(args.output),
                    "sample_rate": sample_rate,
                    "total_seconds": total_seconds,
                    "sentences": timestamps
                }

                # Ensure parent directories exist
                if os.path.dirname(args.json_subtitle) != "":
                    os.makedirs(os.path.dirname(args.json_subtitle), exist_ok=True)
                with open(args.json_subtitle, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f">> JSON subtitle saved to: {args.json_subtitle}")

    except Exception as e:
        print(f"\n>> Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
