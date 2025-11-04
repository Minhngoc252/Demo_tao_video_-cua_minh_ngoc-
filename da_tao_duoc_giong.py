import os
from cached_path import cached_path
import soundfile as sf

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
)

# === Hàm chuẩn hoá văn bản (thay vinorm) ===
def simple_vinorm(text: str) -> str:
    text = text.lower().strip()
    # Chuyển số thành chữ cơ bản
    num_map = {
        "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
        "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
    }
    for k, v in num_map.items():
        text = text.replace(k, v)
    # Xử lý dấu câu và khoảng trắng
    text = text.replace("...", ".").replace("..", ".")
    text = text.replace(",,", ",").replace(" .", ".").replace(" ,", ",")
    text = text.replace('"', '').replace("“", "").replace("”", "")
    return " ".join(text.split())

# === Hàm xử lý hậu kỳ ===
def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace(" , , ", " , ").replace(" ,, ", " , ")
    text = text.replace('"', "").replace("“", "").replace("”", "")
    return " ".join(text.split())

# === 1️⃣ Nạp model & vocoder ===
print("🔹 Đang tải mô hình và vocoder... (chờ vài phút nếu lần đầu chạy)")
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
    vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
)

# === 2️⃣ Đường dẫn ===
ref_audio_path = r"D:\WorkSpace\Demo_tao_video\viet_finetune\wavs\mau1.wav"
output_audio_path = r"D:\WorkSpace\Demo_tao_video\viet_finetune\output\output.wav"

text = "Xin chào, đây là giọng nói được tạo tự động bằng mô hình F5 TTS tiếng Việt. Số 123 sẽ được đọc thành một hai ba."

# === 3️⃣ Tiền xử lý ===
print("🔹 Đang xử lý giọng mẫu...")
# ✅ Bỏ PhoWhisper, dùng trực tiếp text mẫu
ref_audio, _ = preprocess_ref_audio_text(
    ref_audio_path,
    "xin chào, tôi là giọng đọc mẫu dành cho huấn luyện mô hình tiếng việt."
)

# Chuẩn hoá văn bản đầu vào
text_norm = post_process(simple_vinorm(text)).lower()

# === 4️⃣ Tạo giọng nói ===
print("🎙️ Đang tạo giọng nói...")
final_wave, final_sample_rate, _ = infer_process(
    ref_audio, 
    "xin chào, tôi là giọng đọc mẫu dành cho huấn luyện mô hình tiếng việt.", 
    text_norm, 
    model, 
    vocoder, 
    speed=1.0
)

# === 5️⃣ Xuất file âm thanh ===
os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
sf.write(output_audio_path, final_wave, final_sample_rate)
print(f"✅ Hoàn tất! File đã được lưu tại: {os.path.abspath(output_audio_path)}")
