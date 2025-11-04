from f5_tts.api import F5TTS
import soundfile as sf
import numpy as np
import os

# === 1. Khởi tạo model ===
tts = F5TTS(device="cuda")

# === 2. Cấu hình mẫu tham chiếu ===
ref_file = r"D:\WorkSpace\Demo tao video\viet_finetune\wavs\mau1.wav"
ref_text = "Xin chào, tôi là giọng đọc mẫu dành cho huấn luyện mô hình tiếng Việt."

# === 3. Văn bản cần sinh ===
gen_text = "Chào mừng bạn quay trở lại! Hôm nay chúng ta sẽ cùng nhau tìm hiểu về công nghệ trí tuệ nhân tạo."

# === 4. Gọi infer() để sinh audio ===
audio_segments = tts.infer(
    ref_file=ref_file,
    ref_text=ref_text,
    gen_text=gen_text,
    speed=1.0,
    target_rms=0.1,
    cfg_strength=2,
)

# === 5. Debug kết quả trả về ===
print("\nDEBUG: kiểu và kích thước các phần tử được trả về:")
print([type(x) for x in audio_segments])
print([None if not isinstance(x, np.ndarray) else x.shape for x in audio_segments])

# === 6. Lấy phần tử audio 1D ===
audio = None
for seg in audio_segments:
    if isinstance(seg, np.ndarray) and seg.ndim == 1:
        audio = seg
        break

if audio is None:
    raise RuntimeError("❌ Không tìm thấy đoạn audio hợp lệ trong kết quả infer().")

# === 7. Lưu ra file .wav ===
sf.write("output.wav", audio, samplerate=24000)
print("✅ Đã lưu file output.wav tại:", os.path.abspath("output.wav"))
