import os
import re
import subprocess
from moviepy.editor import AudioFileClip, concatenate_audioclips, CompositeAudioClip, vfx

# ====== CẤU HÌNH THƯ MỤC ======
BASE_DIR = r"D:\WorkSpace\Demo tao video"
IMAGE_FOLDER = os.path.join(BASE_DIR, "Anh lam video")
MUSIC_FOLDER = os.path.join(BASE_DIR, "Music")
AUDIO_FOLDER = os.path.join(BASE_DIR, "Audio")
OUTRO_PATH = os.path.join(BASE_DIR, "Outro - Copy.mp4")
OUTPUT_FILE = os.path.join(BASE_DIR, "video_truyen.mp4")
TEMP_AUDIO = os.path.join(BASE_DIR, "temp_audio_mix.mp3")

# ====== CẤU HÌNH VIDEO / AUDIO ======
IMAGE_DURATION = 30
MOVE_START = 3000
MOVE_END = -3000
FPS = 30
ZOOM = 4.2
RESOLUTION = (1920, 1080)
BITRATE = "4000k"
TRANSITION_DURATION = 0.4
STORY_VOLUME_DB = 6
STORY_SPEED = 1.4
MUSIC_VOLUME_DB = -27

# ====== HÀM HỖ TRỢ ======
def db_to_gain(db):
    return 10 ** (db / 20)

def natural_key(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else float("inf")

def load_files(folder, exts):
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and f.lower().endswith(exts):
            files.append(path)
    files.sort(key=lambda x: os.path.basename(x).lower())
    return files

# ====== TÌM FILE ======
musics = load_files(MUSIC_FOLDER, (".mp3", ".wav"))
images = load_files(IMAGE_FOLDER, (".jpg", ".jpeg", ".png"))

if not musics:
    raise FileNotFoundError("Không tìm thấy file nhạc trong thư mục Music!")
if not images:
    raise FileNotFoundError("Không tìm thấy ảnh trong thư mục Anh lam video!")

print(f"🎵 Số bài nhạc nền: {len(musics)}")
print(f"🖼️ Số ảnh: {len(images)}")

# ====== GHÉP AUDIO TRUYỆN ======
audio_chaps = []
for folder in sorted(os.listdir(AUDIO_FOLDER), key=natural_key):
    folder_path = os.path.join(AUDIO_FOLDER, folder)
    if os.path.isdir(folder_path):
        mp3_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path), key=natural_key)
            if f.lower().endswith(".mp3")
        ]
        if mp3_files:
            clips = [AudioFileClip(m) for m in mp3_files]
            audio_chaps.append(concatenate_audioclips(clips))

if not audio_chaps:
    raise FileNotFoundError("Không tìm thấy file .mp3 trong thư mục Audio!")

story_audio = concatenate_audioclips(audio_chaps)
story_audio = story_audio.fx(vfx.speedx, STORY_SPEED).volumex(db_to_gain(STORY_VOLUME_DB))
print(f"📖 Audio truyện dài: {story_audio.duration/60:.1f} phút")

# ====== GHÉP NHẠC NỀN ======
bg_clips = [AudioFileClip(m) for m in musics]
bg_audio = concatenate_audioclips(bg_clips)
while bg_audio.duration < story_audio.duration:
    bg_audio = concatenate_audioclips([bg_audio] + bg_clips)
bg_audio = bg_audio.subclip(0, story_audio.duration).volumex(db_to_gain(MUSIC_VOLUME_DB))

# ====== TRỘN 2 LỚP ÂM THANH ======
final_audio = CompositeAudioClip([bg_audio, story_audio])
final_audio.write_audiofile(TEMP_AUDIO, fps=44100, bitrate="192k")

# ====== TẠO DANH SÁCH ẢNH CHO FFmpeg ======
ffmpeg_inputs = []
for img in images:
    ffmpeg_inputs += ["-loop", "1", "-t", str(IMAGE_DURATION), "-i", img]

# ====== FILTER COMPLEX CHUẨN CHO FFmpeg 8 ======
frames = IMAGE_DURATION * FPS
filter_complex = ""
delta = MOVE_START - MOVE_END

for i in range(len(images)):
    # ⚠️ Sử dụng 'on' thay cho 'n' (chuẩn FFmpeg 8)
    filter_complex += (
        f"[{i}:v]scale=-1:{int(RESOLUTION[1]*ZOOM)},"
        f"crop={RESOLUTION[0]}:{RESOLUTION[1]},"
        f"zoompan=z='1':x='0':y='({MOVE_START})-({delta}*(on/{frames}))':"
        f"d={frames}:fps={FPS},format=yuv420p[v{i}];"
    )

# ====== Chuyển cảnh mượt ======
for i in range(len(images)):
    if i == 0:
        filter_complex += f"[v0]trim=duration={IMAGE_DURATION}[vout0];"
    else:
        prev = f"vout{i-1}"
        offset = IMAGE_DURATION * i - TRANSITION_DURATION
        filter_complex += (
            f"[{prev}][v{i}]xfade=transition=fade:duration={TRANSITION_DURATION}:"
            f"offset={offset}[vout{i}];"
        )

filter_complex += f"[vout{len(images)-1}]format=yuv420p[vfinal];"

# ====== RENDER VIDEO ======
cmd = [
    "ffmpeg", "-y",
    *ffmpeg_inputs,
    "-i", TEMP_AUDIO,
    "-filter_complex", filter_complex,
    "-map", "[vfinal]",
    f"-map", f"{len(images)}:a",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-b:v", BITRATE,
    "-c:a", "aac",
    "-shortest",
    OUTPUT_FILE
]

print("🚀 Đang render video có hiệu ứng pan dọc + chuyển cảnh mượt (FFmpeg 8)...")
subprocess.run(cmd, check=True)
print("✅ Video chính hoàn tất!")

# ====== GHÉP OUTRO ======
if os.path.exists(OUTRO_PATH):
    concat_list = os.path.join(BASE_DIR, "concat_list.txt")
    final_with_outro = OUTPUT_FILE.replace(".mp4", "_final.mp4")
    with open(concat_list, "w", encoding="utf-8") as f:
        f.write(f"file '{OUTPUT_FILE}'\n")
        f.write(f"file '{OUTRO_PATH}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c", "copy", final_with_outro
    ], check=True)

    print(f"🎬 Video cuối cùng (có outro): {final_with_outro}")
else:
    print(f"🎬 Video cuối cùng: {OUTPUT_FILE}")
