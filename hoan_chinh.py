import os
import subprocess
import time

# === Danh sách các script cần chạy lần lượt ===
scripts = [
    "crawl_tu_chuong.py",
    "translate_and_tts.py",
    "Ghep_Audio.py"
]

for script in scripts:
    print(f"\n🚀 Đang chạy: {script}")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy {script}, dừng toàn bộ pipeline.")
        break
    print(f"✅ Hoàn tất {script}\n")
    time.sleep(3)  # nghỉ vài giây giữa các bước

print("🎯 Toàn bộ quy trình crawl → dịch → audio → ghép đã hoàn thành!")
