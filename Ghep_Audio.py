#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge WAV audio by chapter using temp files (optimized for MP3 output & low RAM).
- Input: WAV files.
- Output: MP3 (compressed, small).
- Inserts 0.07s silence between clips.
- Concat WAV chapters by stream copy (fast).
- When exporting a part: concat list -> encode directly to MP3 (no huge WAV temp).
- Splits output parts by MAX_RAW_DURATION.
- Deletes temps to avoid running out of disk/RAM.
- Writes detailed log.
"""

import os
import re
import tempfile
import subprocess
import shutil
from pathlib import Path
import time

# ====== CONFIG ======
BASE_DIR = r"D:\WorkSpace\Demo_tao_video"
AUDIO_FOLDER = os.path.join(BASE_DIR, "Audio")
OUTPUT_FILE_BASE = os.path.join(BASE_DIR, "output_audio_mix")
LOG_FILE = os.path.join(BASE_DIR, "merged_log.txt")

# split parts to avoid extremely large single outputs (seconds)
MAX_RAW_DURATION = 16.2 * 3600  # default 4h per mp3 part — bạn có thể tăng lên (ví dụ 8h, 16h)
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
SILENCE_DURATION = 0.07  # giây
MP3_BITRATE = "128k"     # thể thay đổi 128k / 256k

# ====== HELPERS ======
def natural_key(name: str):
    s = os.path.basename(name).lower()
    tokens, i = [], 0
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[i:j]))
            i = j
        else:
            tokens.append(s[i])
            i += 1
    return tokens

def run_check(cmd, silent=False):
    if not silent:
        print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout

def ffprobe_duration(path):
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        out = run_check(cmd, silent=True).strip()
        return float(out)
    except:
        return 0.0

def concat_list_file(paths, dest, filename="concat_list.txt"):
    """Write a concat list file (with forward slashes). Return its path."""
    list_path = os.path.join(dest, filename)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            p_fixed = p.replace("\\", "/")
            f.write(f"file '{p_fixed}'\n")
    return list_path

# ====== CORE ======
def make_temp_chapter(wav_paths, tmpdir, silence_file, chapter_name):
    """Create 1 temp WAV for chapter by concatenating wav_paths with silence between (stream copy)."""
    if not wav_paths:
        return None, 0.0

    list_with_silence = os.path.join(tmpdir, f"concat_{abs(hash(chapter_name)) & 0xffffffff}.txt")
    with open(list_with_silence, "w", encoding="utf-8") as f:
        for i, p in enumerate(wav_paths):
            f.write(f"file '{p.replace('\\', '/')}'\n")
            if i < len(wav_paths) - 1:
                f.write(f"file '{silence_file.replace('\\', '/')}'\n")

    safe = re.sub(r"[^\w\-_. ]", "_", chapter_name)[:60]
    out_tmp = os.path.join(tmpdir, f"chap_{safe}_{abs(hash(chapter_name)) & 0xffffffff}.wav")

    # concat WAV bằng copy stream => rất nhanh, không giải mã/mã hóa lại
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_with_silence, "-c", "copy", out_tmp]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    dur = ffprobe_duration(out_tmp)
    return out_tmp, dur

def export_part_to_mp3(chapter_temp_files, out_mp3, bitrate=MP3_BITRATE):
    """
    Concat the chapter temp WAVs (by list) and encode directly to MP3.
    This avoids creating a single huge WAV on disk.
    """
    if not chapter_temp_files:
        raise ValueError("No chapter files to export.")

    tmpdir = os.path.dirname(chapter_temp_files[0])
    # chapter_temp_files is list of paths (strings)
    listfile = concat_list_file(chapter_temp_files, tmpdir, filename=f"concat_part_{abs(hash(out_mp3)) & 0xffffffff}.txt")

    # Encode to mp3 directly from concat list (streaming)
    cmd = [
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-vn", "-codec:a", "libmp3lame", "-b:a", bitrate, out_mp3
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ====== MAIN ======
def main():
    start_time = time.time()
    os.makedirs(BASE_DIR, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="chap_tmp_", dir=BASE_DIR)
    log_lines = []
    chapter_temp_files = []   # list of (temp_path, duration, source_folder)
    output_files = []

    # tạo 1 file silence (WAV) duy nhất
    silence_file = os.path.join(tmpdir, f"silence_{int(SILENCE_DURATION*1000)}ms.wav")
    subprocess.run([
        FFMPEG, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(SILENCE_DURATION), "-acodec", "pcm_s16le", silence_file
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # danh sách chapter
    chapters = [d for d in sorted(os.listdir(AUDIO_FOLDER), key=natural_key)
                if os.path.isdir(os.path.join(AUDIO_FOLDER, d))]
    print(f"Found {len(chapters)} chapter folders.")
    log_lines.append(f"Found {len(chapters)} chapter folders.")

    # Tạo temp WAV per-chapter (stream copy)
    for chap in chapters:
        chap_dir = os.path.join(AUDIO_FOLDER, chap)
        wavs = [os.path.join(chap_dir, f) for f in sorted(os.listdir(chap_dir), key=natural_key)
                if f.lower().endswith(".wav")]
        if not wavs:
            log_lines.append(f"Skipped (no wav): {chap}")
            print(f"Skipped (no wav): {chap}")
            continue
        try:
            temp_file, dur = make_temp_chapter(wavs, tmpdir, silence_file, chap)
            chapter_temp_files.append((temp_file, dur, chap_dir))
            log_lines.append(f"Chapter {chap} -> {temp_file} ({dur/60:.2f} min)")
            print(f"Created chapter: {chap} ({dur/60:.2f} min)")
        except Exception as e:
            log_lines.append(f"ERROR creating temp for {chap}: {e}")
            print(f"ERROR creating temp for {chap}: {e}")

    # Ghép thành các MP3 part: concat list -> encode MP3 trực tiếp, sau đó xóa WAV temps đã dùng
    current_list, current_folders = [], []
    current_raw_duration, part_idx = 0.0, 1

    # helper nhỏ: chuyển chapter_temp_files tuples -> path list when needed
    for temp_path, dur, src_folder in chapter_temp_files:
        if current_raw_duration + dur > MAX_RAW_DURATION and current_list:
            mp3_out = f"{OUTPUT_FILE_BASE}_part{part_idx:02d}.mp3"
            print(f"Exporting part {part_idx} -> {mp3_out} ...")
            try:
                export_part_to_mp3(current_list, mp3_out, bitrate=MP3_BITRATE)
                output_files.append(mp3_out)
                # xóa các temp chapter đã ghép
                for tf in current_list:
                    try:
                        os.remove(tf)
                    except OSError:
                        pass
                log_lines.append(f"Exported {mp3_out} ({current_raw_duration/3600:.2f} h)")
            except Exception as e:
                log_lines.append(f"ERROR exporting part {part_idx}: {e}")
                print(f"ERROR exporting part {part_idx}: {e}")
            part_idx += 1
            current_list, current_folders, current_raw_duration = [], [], 0.0

        current_list.append(temp_path)
        current_folders.append(src_folder)
        current_raw_duration += dur

    # final part
    if current_list:
        mp3_out = f"{OUTPUT_FILE_BASE}_part{part_idx:02d}.mp3"
        print(f"Exporting final part {part_idx} -> {mp3_out} ...")
        try:
            export_part_to_mp3(current_list, mp3_out, bitrate=MP3_BITRATE)
            output_files.append(mp3_out)
            for tf in current_list:
                try:
                    os.remove(tf)
                except OSError:
                    pass
            log_lines.append(f"Exported {mp3_out} ({current_raw_duration/3600:.2f} h)")
        except Exception as e:
            log_lines.append(f"ERROR final export: {e}")
            print(f"ERROR final export: {e}")

    # Cleanup tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)

    total_time = time.time() - start_time
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== MERGE AUDIO LOG (Output MP3, optimized) ===\n\n")
        f.write("\n".join(log_lines))
        f.write(f"\n\nTổng thời gian: {total_time/60:.1f} phút\n")
        f.write("\n---\nExported parts:\n")
        for i, of in enumerate(output_files, 1):
            try:
                size_mb = os.path.getsize(of) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            f.write(f"Part {i}: {of} ({size_mb:.1f} MB)\n")

    print("✅ Done. Exported MP3 parts:", output_files)
    print("📄 Log:", LOG_FILE)

if __name__ == "__main__":
    main()
