import os
import subprocess
from tqdm import tqdm  # hiển thị tiến trình

# ====== CẤU HÌNH ======
input_dir = r"D:\WorkSpace\Demo tao video\Reset Output"
output_dir = r"D:\WorkSpace\Demo tao video\viet_finetune\wavs"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# ====== HÀM CHUYỂN ĐỔI ======
def convert_mp3_to_wav(input_path, output_path):
    """
    Chuyển MP3 sang WAV (22.05kHz, mono, PCM 16-bit) — chuẩn cho huấn luyện TTS.
    """
    command = [
        "ffmpeg",
        "-y",                  # ghi đè file cũ
        "-i", input_path,      # file đầu vào
        "-ar", "22050",        # tần số mẫu (Hz)
        "-ac", "1",            # 1 kênh (mono)
        "-acodec", "pcm_s16le",# định dạng 16-bit PCM
        "-map_metadata", "-1", # xóa metadata
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ====== LẤY DANH SÁCH FILE ======
mp3_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp3")]

if not mp3_files:
    print("⚠️ Không tìm thấy file .mp3 nào trong thư mục đầu vào.")
else:
    print(f"🔍 Tìm thấy {len(mp3_files)} file cần xử lý.\n")

    # ====== VÒNG LẶP CHUYỂN ĐỔI CÓ THANH TIẾN TRÌNH ======
    for file in tqdm(mp3_files, desc="🎧 Đang chuyển đổi", unit="file"):
        input_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        output_path = os.path.join(output_dir, base_name + ".wav")

        convert_mp3_to_wav(input_path, output_path)

    print(f"\n✅ Hoàn tất! Tổng số file đã xử lý: {len(mp3_files)}")
    print(f"📂 File WAV chuẩn TTS nằm trong: {output_dir}")
