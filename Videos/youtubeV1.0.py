#!/usr/bin/env python3
import os
import sys
import csv
import re
import shutil
import time
import random
import logging
from pathlib import Path
from typing import List, Set, Tuple, Optional
import yt_dlp
from yt_dlp.postprocessor import PostProcessor
from google.cloud import storage
from dataclasses import dataclass, field

@dataclass
class Config:
    CSV_FILE: Path = Path("videos_statistics.csv")
    ARCHIVE_FILE: Path = Path("downloaded_videos.txt")
    OUTPUT_DIR: Path = Path.home() / "Downloads" / "YouTube"
    BUCKET_NAME: str = "b2b-juanguerra"
    UPLOAD_TO_GCS: bool = True
    CLEANUP_AFTER_UPLOAD: bool = True
    AUDIO_ONLY: bool = False
    QUALITY: str = "worst"
    FORCE_REDOWNLOAD: bool = False
    MAX_VIDEOS: int = 25000
    VISITOR_DATA: Optional[str] = None
    COOKIES_FILES: List[Path] = field(default_factory=lambda: [Path("co_cookies.txt"), Path("ec_cookies.txt"), Path("ru_cookies.txt"), Path("zh_cookies.txt"), Path("hk_cookies.txt")])
    COOKIES_FROM_BROWSER: Optional[str] = "chrome"
    CPU_COUNT: int = random.randint(4, 8)
    CONCURRENT_FRAGMENTS: int = random.randint(4, 16)
    MIN_FREE_SPACE_GB: float = 0.5
    BATCH_CHECK_SIZE: int = 1000
    BATCH_DOWNLOAD_SIZE: int = random.randint(4, 16)
config = Config()

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def check_disk_space() -> bool:
    free_gb = shutil.disk_usage(config.OUTPUT_DIR).free / (1024**3)
    if free_gb < config.MIN_FREE_SPACE_GB:
        log.warning(f"Low disk space: {free_gb:.1f} GB free")
        return free_gb >= 0.2
    return True


def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/v/([^&\n?#]+)',
    ]
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return url.split('/')[-1].split('?')[0]

def normalize_youtube_url(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if 'youtube.com' in url_or_id or 'youtu.be' in url_or_id:
        return url_or_id
    return f"https://www.youtube.com/watch?v={url_or_id}"

class GCSUploader(PostProcessor):
    def __init__(self, bucket_name: str, cleanup: bool):
        super().__init__()
        if storage is None:
            log.warning("google.cloud.storage not available; GCS upload disabled")
            self.storage_client = None
            self.bucket = None
        else:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(bucket_name)
            except Exception as e:
                log.error(f"GCS init failed: {e}")
                self.storage_client = None
                self.bucket = None
        self.cleanup = cleanup

    def run(self, info):
        if not self.storage_client or self.bucket is None:
            return [], info

        if not isinstance(info, dict):
            log.error(f"Expected dict but got {type(info)}: {info}")
            return [], info
 
        # Ensure filepath is properly accessed from the info dictionary
        filepath = info.get('filepath')
        if not filepath:
            log.error("No filepath found in info")
            return [], info
 
        local_path = Path(filepath)
        if not local_path.exists():
            log.warning(f"File not found: {local_path}")
            return [], info

        blob_name = local_path.name
        if blob_name.startswith('_'):
            blob_name = f"file{blob_name}"
            log.info(f"Renamed file starting with underscore: {local_path.name} -> {blob_name}")
            
        blob = self.bucket.blob(blob_name)
        log.info(f"GCS Upload: {blob_name}")

        try:
            blob.upload_from_filename(str(local_path))
            log.info(f"Upload complete: {local_path.name}")
            # Only return the file for cleanup if upload was successful AND cleanup is enabled
            return ([str(local_path)] if self.cleanup else [], info)
        except Exception as e:
            log.error(f"Upload failed: {e}")
            # Don't return the file for cleanup if upload failed
            return [], info

def get_existing_videos(bucket_name: str, audio_only: bool) -> Set[str]:
    extensions = ['.mp3'] if audio_only else ['.mp4', '.mkv', '.webm', '.avi', '.mov']
    existing_ids = set()

    try:
        if storage is None:
            log.warning("google.cloud.storage not available; skipping existing GCS check")
            return existing_ids

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        log.info("Checking existing GCS files...")

        page_token = None
        while True:
            for blob in bucket.list_blobs(page_size=config.BATCH_CHECK_SIZE, page_token=page_token):
                if any(blob.name.lower().endswith(ext) for ext in extensions):
                    video_id = Path(blob.name).stem.split(' - ')[0]
                    existing_ids.add(video_id)
            if not page_token:
                break

        log.info(f"Found {len(existing_ids)} existing videos.")
    except Exception as e:
        log.warning(f"Could not check GCS: {e}")

    return existing_ids

def build_yt_dlp_opts() -> dict:
    opts = {
        "outtmpl": str(config.OUTPUT_DIR / "%(id)s.%(ext)s"),
        "format_sort": ["+size", "+br", "+res", "+fps"],
        "concurrent_fragments": min(config.CONCURRENT_FRAGMENTS, 8),
        "sleep_interval": 2,
        "max_sleep_interval": 4,
        "retries": 3,
        "fragment_retries": 5,
        "quiet": True,
        "no_warnings": True,
        "download_archive": str(config.ARCHIVE_FILE),
    }
    
    client_to_emulate = random.choice(["web", "android", "mweb", "ios", "tv", "web_creator", "web_safari"])
    log.info(f"Using '{client_to_emulate}' client for this batch.")

    opts["extractor_args"] = {
        "youtube": {
            "innertube_client": client_to_emulate,
        }
    }

    if config.VISITOR_DATA:
        log.info("Using Visitor Data configuration.")
        opts["extractor_args"]["youtube"]["visitor_data"] = config.VISITOR_DATA

    valid_cookie_files = [f for f in config.COOKIES_FILES if f.exists()]
    if valid_cookie_files:
        cookie_file_to_use = random.choice(valid_cookie_files)
        opts["cookiefile"] = str(cookie_file_to_use)
        log.info(f"Using cookies from file: {cookie_file_to_use}")

    if "cookiefile" not in opts and config.COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.COOKIES_FROM_BROWSER,)
        log.info(f"Using cookies from browser: {config.COOKIES_FROM_BROWSER}")
    elif "cookiefile" not in opts and not config.VISITOR_DATA:
        log.warning(
            "No Visitor Data or cookies configured — you may encounter bot detection. "
            "To use cookies, set COOKIES_FILE or COOKIES_FROM_BROWSER. "
            "To use visitor data, set VISITOR_DATA. "
            "You can get visitor data from your browser's network requests to YouTube."
        )

    return opts
def cleanup_download_folder():
    if config.OUTPUT_DIR.exists():
        for file_path in config.OUTPUT_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                log.info(f"Deleted: {file_path.name}")
        log.info(f"Cleaned up download folder: {config.OUTPUT_DIR}")

def download_batch(urls: List[str]) -> Tuple[int, int]:
    if not urls:
        return 0, 0

    ydl_opts = build_yt_dlp_opts()
    success, fail = 0, 0
    upload_failed = False

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if config.UPLOAD_TO_GCS and storage is not None:
            # Create uploader but don't enable automatic cleanup
            # We'll handle cleanup separately based on upload success
            uploader = GCSUploader(config.BUCKET_NAME, False)
            ydl.add_post_processor(uploader, when='after_move')
        for url in urls:
            if not check_disk_space():
                log.error("Insufficient disk space. Stopping.")
                break
            try:
                normalized_url = normalize_youtube_url(url)
                video_id = extract_video_id(normalized_url)
                log.info(f"Downloading: {video_id}")
                result = ydl.download([normalized_url])
                if result != 0:
                    upload_failed = True
                    log.warning(f"Upload may have failed for {video_id}")
                success += 1
                time.sleep(random.uniform(0.25, 0.75))
            except Exception as e:
                log.error(f"Failed {extract_video_id(url)}: {e}")
                upload_failed = True
                fail += 1

    # Only clean up if uploads were successful and cleanup is enabled
    if config.CLEANUP_AFTER_UPLOAD and not upload_failed:
        log.info("All uploads successful, cleaning up download folder")
        cleanup_download_folder()
    elif upload_failed:
        log.warning("Some uploads failed, keeping downloaded files")

    return success, fail

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube downloader with optional GCS upload")
    parser.add_argument('--dry-run', action='store_true', help='List videos and exit without downloading')
    parser.add_argument('--no-gcs', action='store_true', help='Disable Google Cloud Storage upload')
    args = parser.parse_args()

    if args.no_gcs:
        config.UPLOAD_TO_GCS = False


    if not config.CSV_FILE.exists():
        log.error(f"CSV file not found: {config.CSV_FILE}")
        sys.exit(1)


    try:
        with config.CSV_FILE.open(newline='', encoding='utf-8') as f:
            raw_urls = [row['video_url'] for row in csv.DictReader(f) if row.get('video_url')]
            random.shuffle(raw_urls)
            urls = [normalize_youtube_url(url) for url in raw_urls]
    except Exception as e:
        log.error(f"CSV read error: {e}")
        sys.exit(1)

    if not urls:
        log.info("No URLs found.")
        return

    if args.dry_run:
        log.info("Dry run: listing first 20 normalized URLs")
        for u in urls[:20]:
            log.info(u)
        return

    if config.MAX_VIDEOS and len(urls) > config.MAX_VIDEOS:
        urls = urls[:config.MAX_VIDEOS]
        log.info(f"Limited to {len(urls)} videos.")

    if not config.FORCE_REDOWNLOAD:
        # yt-dlp will handle skipping videos in the archive file
        log.info(f"Using archive file at {config.ARCHIVE_FILE} to skip already downloaded videos.")
    else:
        log.info("FORCE_REDOWNLOAD is True, ignoring archive file.")

    total_success, total_fail = 0, 0
    start_time = time.time()

    i = 0
    batch_num = 0
    while i < len(urls):
        batch_num += 1
        # Generate a new random batch size for each iteration
        current_batch_size = random.randint(4, 16)
        batch = urls[i:i + current_batch_size]
        log.info(f"=== Batch {batch_num} ({len(batch)} videos) ===")
        success, fail = download_batch(batch)
        total_success += success
        total_fail += fail
        i += len(batch)  # Move to the next starting index
        elapsed = time.time() - start_time
        avg_time = elapsed / max(total_success + total_fail, 1)
        log.info(f"Batch done: {success} success, {fail} failed (avg {avg_time:.1f}s/video)")

    log.info("=== DOWNLOAD COMPLETE ===")
    log.info(f"Total: {total_success} success, {total_fail} failed")
    log.info(f"Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
