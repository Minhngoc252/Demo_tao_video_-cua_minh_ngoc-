import asyncio
import os
import re
import random
from datetime import datetime
from urllib.parse import urljoin

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup

# ====================================================
# 🔧 CẤU HÌNH
# ====================================================
INPUT_FILE = "input.txt"   # file chứa mã HTML copy từ trang
RATE_LIMIT = 10            # request mỗi phút (giới hạn để tránh bị chặn IP)
DELAY = 60 / RATE_LIMIT    # thời gian delay giữa các chương
MAX_RETRIES = 5            # số lần thử lại khi lỗi kết nối
OUTPUT_FOLDER = "output"   # nơi lưu chương
# ====================================================


def natural_key(name: str):
    """Dùng để sắp xếp chương theo thứ tự tự nhiên (1, 2, 10 thay vì 1, 10, 2)."""
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ============================
# 🕷️ Đọc danh sách chương từ file input.txt
# ============================
async def get_chapter_list_from_file():
    print("📂 Đang đọc danh sách chương từ file input.txt...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"⚠️ Không tìm thấy file {INPUT_FILE}! Hãy copy mã nguồn trang (Ctrl+U) rồi dán vào file này."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Trên NovelBin, các link chương nằm trong phần class="list-chapter"
    links = soup.select("ul.list-chapter li a")

    chapters = []
    for link in links:
        title = link.get_text(strip=True)
        href = link.get("href")
        if href and title:
            full_url = urljoin("https://novelbin.com", href)
            chapters.append((title, full_url))

    if not chapters:
        raise Exception("⚠️ Không tìm thấy danh sách chương! Có thể bạn copy chưa hết mã HTML trang.")

    # ✅ Sắp xếp chương theo thứ tự tự nhiên
    chapters.sort(key=lambda x: natural_key(x[0]))

    print(f"📚 Tìm thấy {len(chapters)} chương trong file input.txt.")
    return chapters


# ============================
# 📖 Cào nội dung từng chương (có retry + random delay)
# ============================
async def crawl_chapter(crawler, title, url, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        print(f"📘 Đang cào: {title} (lần {attempt})")
        try:
            run_config = CrawlerRunConfig(
                cache_mode="bypass",
                css_selector="div#chr-content, div.text-left",
                exclude_external_links=True
            )

            result = await crawler.arun(url=url, config=run_config, timeout_ms=60000)

            if result.success:
                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                safe_name = re.sub(r'[\\/*?:"<>|]', "_", title)
                file_name = f"{safe_name}.txt"
                file_path = os.path.join(OUTPUT_FOLDER, file_name)

                # Làm sạch nội dung markdown nếu có tag hoặc ký tự lạ
                cleaned = re.sub(r'\s+', ' ', result.markdown.strip())

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"💾 Đã lưu {file_name}")
                return  # ✅ Thành công thì thoát

            else:
                print(f"⚠️ Lỗi khi cào {title} ({result.status_code})")

        except Exception as e:
            print(f"❌ Lỗi khi cào {title}: {e}")

        if attempt < retries:
            wait_time = 8 + random.uniform(1, 5)
            print(f"🔁 Thử lại sau {wait_time:.1f} giây...")
            await asyncio.sleep(wait_time)

    print(f"🚫 Bỏ qua {title} sau {retries} lần thất bại.")


# ============================
# 🏁 Main
# ============================
async def main():
    browser_config = BrowserConfig(
        headless=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        chapters = await get_chapter_list_from_file()

        total = len(chapters)
        for idx, (title, url) in enumerate(chapters, start=1):
            await crawl_chapter(crawler, title, url)

            if idx < total:
                delay_time = DELAY + random.uniform(0.5, 3.0)
                print(f"⏳ Chờ {delay_time:.1f}s trước khi tiếp tục...")
                await asyncio.sleep(delay_time)

    print("🎉 Hoàn tất toàn bộ chương!")


if __name__ == "__main__":
    asyncio.run(main())
