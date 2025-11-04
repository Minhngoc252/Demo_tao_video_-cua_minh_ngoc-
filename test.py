import os
import re
import time
import argparse
import soundfile as sf
import torch
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, load_model, infer_process


# ==============================
# 🔹 HÀM TIỆN ÍCH
# ==============================
def num_to_vn(num: int) -> str:
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    num_str = str(num)
    if len(num_str) == 1:
        return digits[int(num_str)]
    if len(num_str) == 2:
        a, b = int(num_str[0]), int(num_str[1])
        if a == 1:
            if b == 0:
                return "mười"
            elif b == 5:
                return "mười lăm"
            else:
                return f"mười {digits[b]}"
        else:
            if b == 0:
                return f"{digits[a]} mươi"
            elif b == 5:
                return f"{digits[a]} lăm"
            else:
                return f"{digits[a]} {digits[b]}"
    if len(num_str) == 3:
        a, b, c = int(num_str[0]), int(num_str[1]), int(num_str[2])
        res = f"{digits[a]} trăm"
        if b == 0 and c == 0:
            return res
        if b == 0:
            res += " lẻ"
        elif b == 1:
            res += " mười"
        else:
            res += f" {digits[b]} mươi"
        if c != 0:
            if c == 5 and b != 0:
                res += " lăm"
            else:
                res += f" {digits[c]}"
        return res
    if len(num_str) == 4:
        a, b, c, d = [int(x) for x in num_str]
        res = f"{digits[a]} nghìn"
        if b == 0 and c == 0 and d == 0:
            return res
        elif b == 0 and c == 0:
            res += f" lẻ {digits[d]}"
        else:
            res += " " + num_to_vn(int(num_str[1:]))
        return res
    return num_str


def simple_vinorm(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt cho TTS."""
    text = text.strip()
    letter_map = {
        "A": "ây", "B": "bi", "C": "xi", "D": "đi", "E": "i", "F": "ép",
        "G": "gi", "H": "hát", "I": "ai", "J": "gi", "K": "ca", "L": "eo",
        "M": "em", "N": "en", "O": "âu", "P": "pi", "Q": "qui", "R": "a",
        "S": "ét", "T": "ti", "U": "u", "V": "vi", "W": "đúp",
        "X": "ích", "Y": "y", "Z": "dét"
    }

    def replace_abbrev(match):
        abbr = match.group()
        letters = [letter_map.get(ch, ch.lower()) for ch in abbr]
        return " ".join(letters)

    text = re.sub(r"\b[A-Z]{2,}\b", replace_abbrev, text)
    text = re.sub(r"\b\d+\b", lambda m: num_to_vn(int(m.group())), text)
    text = (text.replace("...", ".").replace("..", ".")
                .replace(",,", ",").replace(" .", ".")
                .replace(" ,", ",").replace('"', "")
                .replace("“", "").replace("”", ""))
    return " ".join(text.lower().split())


def post_process(text: str) -> str:
    """Làm sạch hậu kỳ văn bản."""
    text = " " + text + " "
    text = (text.replace(" . . ", " . ").replace(" .. ", " . ")
                .replace(" , , ", " , ").replace(" ,, ", " , ")
                .replace('"', "").replace("“", "").replace("”", ""))
    return " ".join(text.split())


# ==============================
# 🔹 CHƯƠNG TRÌNH CHÍNH
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Văn bản cần đọc")
    parser.add_argument("--out", type=str, required=True, help="Đường dẫn file đầu ra (.wav)")
    parser.add_argument("--ref", type=str, default=r"D:\WorkSpace\Demo_tao_video\viet_finetune\wavs\mau1.wav",
                        help="Đường dẫn âm thanh mẫu (đã xử lý sẵn hoặc có cache)")
    parser.add_argument("--speed", type=float, default=0.8, help="Tốc độ đọc (0.8 = chậm hơn 20%)")
    parser.add_argument("--no_ref_process", action="store_true",
                        help="Bỏ qua bước xử lý ref_audio (sử dụng cache có sẵn)")
    args = parser.parse_args()

    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Dùng thiết bị: {device}")

    # Tải mô hình
    print("🔹 Đang tải mô hình F5-TTS và vocoder...")
    vocoder = load_vocoder()
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
        vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
    )
    model.to(device)

    # ==============================
    # ⚡ XỬ LÝ REF AUDIO / CACHE
    # ==============================
    ref_cache = os.path.splitext(args.ref)[0] + "_cache.pt"
    ref_text_default = "xin chào, tôi là giọng đọc mẫu dành cho huấn luyện mô hình tiếng việt."

    if os.path.exists(ref_cache):
        print(f"⚡ Dùng cache có sẵn: {ref_cache}")
        cache = torch.load(ref_cache, map_location=device)
        ref_audio = cache.get("ref_audio", None)
        ref_text = cache.get("ref_text", ref_text_default)
    else:
        print("🎧 Đang xử lý ref_audio lần đầu (sẽ lưu cache để dùng lại)...")
        ref_audio, _ = preprocess_ref_audio_text(
            args.ref,
            ref_text_default,
            clip_short=True,
            show_info=print,
            device=device
        )
        torch.save({"ref_audio": ref_audio, "ref_text": ref_text_default}, ref_cache)
        ref_text = ref_text_default
        print(f"💾 Đã lưu cache: {ref_cache}")

    # ==============================
    # 🔊 SINH GIỌNG NÓI
    # ==============================
    text_norm = post_process(simple_vinorm(args.text)).lower()
    print(f"🎙️ Đang tạo giọng nói cho: {args.out}")
    start_time = time.time()
    final_wave, final_sr, _ = infer_process(
        ref_audio, ref_text, text_norm, model, vocoder, speed=args.speed
    )
    end_time = time.time()
    print(f"🕒 Thời gian tạo: {end_time - start_time:.2f}s")

    # Lưu file WAV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, final_wave, final_sr)
    print(f"💾 Đã lưu: {args.out}")
    print("✅ Hoàn tất.")
