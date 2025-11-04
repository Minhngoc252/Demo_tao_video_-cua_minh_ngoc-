import os
import subprocess

# ====== CONFIG ======
BASE_DIR = r"D:\WorkSpace\Demo tao video"
RESET_FOLDER = os.path.join(BASE_DIR, "Reset Speed")   # chứa các file output cần chỉnh lại tốc độ
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Reset Output") # nơi lưu file đã khôi phục tốc độ
FFMPEG = "ffmpeg"

# hệ số tốc độ ngược
SPEED_REVERSE = 1 / 1.4   # = 0.7142857

# tăng hoặc giữ nguyên âm lượng
VOLUME_DB = 0  # hoặc -6 nếu bạn muốn giảm lại đúng như trước

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for f in os.listdir(RESET_FOLDER):
    if not f.lower().endswith(".mp3"):
        continue
    src = os.path.join(RESET_FOLDER, f)
    out = os.path.join(OUTPUT_FOLDER, f)
    cmd = [
        FFMPEG, "-y", "-i", src,
        "-filter:a", f"atempo={SPEED_REVERSE},volume={VOLUME_DB}dB",
        "-c:a", "libmp3lame", "-b:a", "192k",
        out
    ]
    print("Restoring:", f)
    subprocess.run(cmd, check=True)

print("✅ Hoàn tất khôi phục tốc độ. File xuất ở:", OUTPUT_FOLDER)
