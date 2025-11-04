import os
import re
import time
import random
import requests
import subprocess
from tqdm import tqdm

# ========= CONFIG =========
API_KEY = "AIzaSyDMKwfXW9JDpC1HKhD1ZdyohH_aVCPWVXo"
MODEL = "models/gemini-2.0-flash-lite"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={API_KEY}"

INPUT_FOLDER = "output"
OUTPUT_TRANSLATED = "output_vi"
AUDIO_FOLDER = "Audio"

MAX_TTS_CHARS = 3800
SLEEP_BETWEEN_CALLS = 10
MAX_RETRIES = 6

PROMPT = """Bạn là một dịch giả tiểu thuyết chuyên nghiệp, có khả năng chuyển ngữ tiếng Anh sang tiếng Việt một cách mượt mà, giàu cảm xúc và tự nhiên như văn học gốc.

Đầu vào có thể là một chương truyện tiếng Anh được sao chép từ trang web, có thể chứa tiêu đề, số chương, bình luận, quảng cáo, hoặc hình ảnh.

Yêu cầu xuất đầu ra:
Chỉ dịch phần nội dung truyện thực sự – bỏ qua tiêu đề, số chương, bình luận, ghi chú, quảng cáo, watermark hoặc các đoạn ngoài truyện.
Dịch sang tiếng Việt một cách tự nhiên, trôi chảy, đúng phong cách của truyện võ hiệp.
Chỉ giữ nguyên tên riêng của nhân vật bằng tiếng Anh.
Không chèn ký tự xuống dòng \\n hoặc dòng trống – toàn bộ văn bản phải là một đoạn liền mạch.
Không thêm tiêu đề, không viết “Chương ...”, không thêm lời mở đầu hay kết luận.
Không thêm chú thích, không mở ngoặc giải thích, không thay đổi định dạng hoặc phông chữ.
Chỉ trả về văn bản tiếng Việt hoàn chỉnh, không kèm hướng dẫn, nhận xét hoặc phần tiếng Anh gốc.
"""

# ========= ĐƯỜNG DẪN F5-TTS =========
F5_TTS_PYTHON = r"D:\WorkSpace\Demo_tao_video\F5-TTS-Vietnamese-100h\env\Scripts\python.exe"
F5_TTS_SCRIPT = r"D:\WorkSpace\Demo_tao_video\F5-TTS-Vietnamese-100h\test.py"

# ⚡ Dùng cache giọng đọc sẵn (đã xử lý ref_audio ở lần đầu)
REF_AUDIO = r"D:\WorkSpace\Demo_tao_video\viet_finetune\wavs\mau1.wav"

# ==========================

def natural_key(name: str):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def call_gemini_api(text: str) -> str:
    data = {
        "contents": [
            {"parts": [
                {"text": PROMPT},
                {"text": text}
            ]}
        ]
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.post(BASE_URL, json=data, timeout=150)
            if res.status_code == 200:
                out = res.json()
                delay = SLEEP_BETWEEN_CALLS + random.uniform(2.0, 5.0)
                print(f"✅ Thành công! Nghỉ {delay:.1f}s trước request tiếp theo...\n")
                time.sleep(delay)
                return out["candidates"][0]["content"]["parts"][0]["text"].strip()

            print(f"⚠️ API Error {res.status_code}: {res.text[:200]}")
            time.sleep(10 * attempt)

        except Exception as e:
            print(f"❌ Lỗi kết nối Gemini ({attempt}/{MAX_RETRIES}): {e}")
            time.sleep(15 * attempt)

    print("🚫 Hết số lần retry, bỏ qua đoạn này.\n")
    return ""


def split_text_for_tts(text: str, max_chars=MAX_TTS_CHARS):
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.findall(r'[^.!?]+[.!?…]*\s*', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"🔊 Văn bản được chia thành {len(chunks)} phần (≤ {max_chars} ký tự).")
    return chunks


# === SỬ DỤNG F5-TTS ĐỂ TẠO GIỌNG ===
def generate_tts_audio(chapter_name: str, text: str):
    parts = split_text_for_tts(text)
    chap_dir = os.path.join(AUDIO_FOLDER, chapter_name)
    os.makedirs(chap_dir, exist_ok=True)

    for i, chunk in enumerate(parts, start=1):
        filename = f"part_{i:03d}.wav"
        output_path = os.path.join(chap_dir, filename)

        if os.path.exists(output_path):
            print(f"⚙️  Bỏ qua (đã có): {output_path}")
            continue

        print(f"🎙️ [{chapter_name}] → {filename}")

        try:
            subprocess.run([
                F5_TTS_PYTHON,
                F5_TTS_SCRIPT,
                "--text", chunk,
                "--out", output_path,
                "--ref", REF_AUDIO,
                "--no_ref_process",  # ⚡ Bỏ qua bước xử lý ref_audio
            ], check=True)

            print(f"✅ Đã tạo: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi tạo F5-TTS cho {filename}: {e}")
            with open("tts_error_log.txt", "a", encoding="utf-8") as logf:
                logf.write(f"{chapter_name}/{filename}\n")


def process_all_chapters():
    os.makedirs(OUTPUT_TRANSLATED, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    txt_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")],
        key=natural_key
    )

    print(f"📚 Tìm thấy {len(txt_files)} chương để xử lý.\n")

    for filename in tqdm(txt_files):
        chapter_name = os.path.splitext(filename)[0]
        chap_path = os.path.join(INPUT_FOLDER, filename)

        with open(chap_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        if not raw_text:
            print(f"⚠️ Bỏ qua {chapter_name} (rỗng).")
            continue

        out_file = os.path.join(OUTPUT_TRANSLATED, filename)
        if os.path.exists(out_file):
            print(f"🈶 Bỏ qua dịch (đã có): {chapter_name}")
            with open(out_file, "r", encoding="utf-8") as f:
                vi_text = f.read().strip()
        else:
            print(f"🈶 Đang dịch chương: {chapter_name} ...")
            vi_text = call_gemini_api(raw_text)
            if not vi_text:
                print(f"⚠️ Không dịch được {chapter_name}, bỏ qua.\n")
                continue
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(vi_text)

        print(f"🎧 Đang tạo audio cho {chapter_name} ...")
        generate_tts_audio(chapter_name, vi_text)

        pause = random.uniform(3.0, 6.0)
        print(f"🕒 Nghỉ {pause:.1f}s trước chương tiếp theo...\n")
        time.sleep(pause)

    print("\n🎉 Hoàn tất dịch và tạo audio cho tất cả chương!")


if __name__ == "__main__":
    process_all_chapters()
