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

def validate_youtube_url(url: str) -> bool:
    patterns = [
        r'youtube\.com/watch\?v=[\w-]+',
        r'youtu\.be/[\w-]+',
        r'youtube\.com/embed/[\w-]+'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def normalize_youtube_url(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if validate_youtube_url(url_or_id):
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
            return ([str(local_path)] if self.cleanup else [], info)
        except Exception as e:
            log.error(f"Upload failed: {e}")
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
        "concurrent_fragments": min(config.CONCURRENT_FRAGMENTS, 16),
        "sleep_interval": get_sleep_time(0.1),
        "max_sleep_interval": get_sleep_time(0.5),
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

    cookie_file = rotate_cookies()
    if cookie_file:
        opts["cookiefile"] = cookie_file
        log.info(f"Using cookies from file: {cookie_file}")
    elif config.COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.COOKIES_FROM_BROWSER,)
        log.info(f"Using cookies from browser: {config.COOKIES_FROM_BROWSER}")
    elif not config.VISITOR_DATA:
        log.warning("No authentication method available. Bot detection likely.")


    return opts

class StatusTracker:
    def __init__(self):
        self.succeeded = set()
        self.failed = set()

    def hook(self, d):
        if d['status'] == 'finished':
            video_id = d.get('info_dict', {}).get('id')
            if video_id:
                self.succeeded.add(video_id)
        elif d['status'] == 'error':
            video_id = d.get('info_dict', {}).get('id')
            if video_id:
                self.failed.add(video_id)
def emergency_cleanup(min_space_gb: float = 1.0):
    free_gb = shutil.disk_usage(config.OUTPUT_DIR).free / (1024**3)
    if free_gb < min_space_gb:
        cleanup_download_folder()
        return True
    return False

def cleanup_download_folder():
    if config.OUTPUT_DIR.exists():
        for file_path in config.OUTPUT_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                log.info(f"Deleted: {file_path.name}")
        log.info(f"Cleaned up download folder: {config.OUTPUT_DIR}")

def get_sleep_time(failure_rate: float) -> float:
    base_sleep = 1.5
    if failure_rate > 0.5:
        return base_sleep * 3
    elif failure_rate > 0.2:
        return base_sleep * 2
    return base_sleep

def rotate_cookies() -> str:
    valid_files = [f for f in config.COOKIES_FILES if f.exists()]
    if not valid_files:
        return None
    return str(random.choice(valid_files))

def retry_failed_downloads(failed_urls: List[str], max_retries: int = 3) -> Tuple[int, int]:
    total_success, total_fail = 0, 0
    current_urls = failed_urls
    
    for attempt in range(max_retries):
        if not current_urls:
            break
        success, fail = download_batch(current_urls)
        total_success += success
        current_urls = current_urls[success:]
        if fail > 0:
            sleep_time = get_sleep_time(fail / len(failed_urls))
            time.sleep(sleep_time)
    
    return total_success, len(current_urls)

def download_batch(urls: List[str]) -> Tuple[int, int]:
    if not urls:
        return 0, 0

    ydl_opts = build_yt_dlp_opts()
    success, fail = 0, 0
    status_tracker = StatusTracker()
    upload_failed = False

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if config.UPLOAD_TO_GCS and storage is not None:
            uploader = GCSUploader(config.BUCKET_NAME, False)
            ydl.add_post_processor(uploader, when='after_move')
        
        ydl.add_progress_hook(status_tracker.hook)

        try:
            ydl.download(urls)
        except Exception as e:
            log.error(f"An error occurred during batch download: {e}")

    success = len(status_tracker.succeeded)
    fail = len(urls) - success

    # Status tracking is now handled by yt-dlp's archive file

    if config.CLEANUP_AFTER_UPLOAD and fail == 0:
        log.info("All uploads successful, cleaning up download folder")
        cleanup_download_folder()
    elif upload_failed:
        log.warning("Some uploads failed, keeping downloaded files")

    return success, fail

def upload_local_files():
    if not config.UPLOAD_TO_GCS or storage is None:
        return

    local_files = list(config.OUTPUT_DIR.glob('*.mp4')) + \
                  list(config.OUTPUT_DIR.glob('*.mkv')) + \
                  list(config.OUTPUT_DIR.glob('*.webm'))

    if not local_files:
        return

    log.info(f"Found {len(local_files)} local file(s) to upload from previous runs.")
    uploader = GCSUploader(config.BUCKET_NAME, cleanup=False)
    if not uploader.storage_client or not uploader.bucket:
        log.error("Cannot upload local files because GCS connection failed.")
        return

    for file_path in local_files:
        log.info(f"Attempting to upload existing local file: {file_path.name}")
        try:
            files_to_delete, _ = uploader.run({'filepath': str(file_path)})
            if files_to_delete:
                file_path.unlink()
                log.info(f"Successfully uploaded and cleaned up {file_path.name}")
        except Exception as e:
            log.error(f"Failed to upload local file {file_path.name}: {e}")



def main():
    upload_local_files()

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

    start_time = time.time()
    total_success = total_fail = 0

    for i in range(0, len(urls), config.BATCH_DOWNLOAD_SIZE):
        if emergency_cleanup():
            log.warning("Emergency cleanup performed")
            
        batch = urls[i:i + config.BATCH_DOWNLOAD_SIZE]
        log.info(f"=== Processing {len(batch)} videos ===")
        
        success, fail = download_batch(batch)
        if fail > 0:
            retry_success, retry_fail = retry_failed_downloads(batch[success:])
            success += retry_success
            fail = retry_fail
            
        total_success += success
        total_fail += fail
        
        avg_time = (time.time() - start_time) / max(total_success + total_fail, 1)
        log.info(f"Progress: {total_success} downloaded, {total_fail} failed")
        log.info(f"Average time per video: {avg_time:.1f}s")

    runtime = (time.time() - start_time) / 60
    log.info(f"=== Download Complete: {total_success} success, {total_fail} failed ===")
    log.info(f"Total runtime: {runtime:.1f} minutes")

if __name__ == "__main__":
    main()
