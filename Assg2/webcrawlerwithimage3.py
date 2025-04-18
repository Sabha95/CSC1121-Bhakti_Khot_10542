import os
import requests
import json
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import uuid
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# BLIP Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Folders
os.makedirs("downloaded_images3", exist_ok=True)

# Config
TARGET_IMAGE_COUNT = 1500
OUTPUT_JSON = "image_metadata_blip8.json"
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "avif"]
SEED_URLS = [
    "https://www.bikewale.com/electric-scooters/"
    "https://www.tvsmotor.com/our-products/vehicles"    
]

def describe_image(img):
    try:
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"BLIP error: {e}")
        return "Could not describe image"

def get_file_extension(url):
    ext = url.split('.')[-1].split('?')[0].lower()
    return ext if ext in SUPPORTED_EXTENSIONS else None

def download_and_process_image(img_url, source_url):
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content)).convert("RGB")
            except UnidentifiedImageError:
                print(f"❌ Unsupported or corrupt image: {img_url}")
                return None

            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("downloaded_images3", filename)
            img.save(filepath, "JPEG")

            caption = describe_image(img)

            return {
                "url": img_url,
                "local_path": filepath,
                "source_url": source_url,
                "vision_caption": caption
            }
    except Exception as e:
        print(f"Failed to download/process image {img_url}: {e}")
    return None

def is_valid_link(link, base_netloc):
    parsed = urlparse(link)
    return (not parsed.netloc or parsed.netloc == base_netloc) and parsed.scheme in ["http", "https", ""]

def crawl_and_scrape(seed_urls):
    collected = []
    seen_imgs = set()
    visited_pages = set()
    url_queue = list(seed_urls)

    while url_queue and len(collected) < TARGET_IMAGE_COUNT:
        current_url = url_queue.pop(0)
        if current_url in visited_pages:
            continue
        visited_pages.add(current_url)

        try:
            print(f"\n🌐 Crawling: {current_url}")
            html = requests.get(current_url, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")

            # Process images
            images = soup.find_all("img")
            for img in tqdm(images, desc="Scraping images"):
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue
                full_url = urljoin(current_url, src)
                if full_url in seen_imgs:
                    continue
                seen_imgs.add(full_url)

                ext = get_file_extension(full_url)
                if not ext:
                    continue

                result = download_and_process_image(full_url, current_url)
                if result:
                    collected.append(result)
                if len(collected) >= TARGET_IMAGE_COUNT:
                    return collected

            # Add internal links to queue
            base_netloc = urlparse(current_url).netloc
            for a in soup.find_all("a", href=True):
                href = a["href"]
                link = urljoin(current_url, href)
                if is_valid_link(link, base_netloc) and link not in visited_pages:
                    url_queue.append(link)

        except Exception as e:
            print(f"⚠️ Error crawling {current_url}: {e}")
            continue

    return collected

# Start crawling
metadata = crawl_and_scrape(SEED_URLS)

# Save metadata
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Done! {len(metadata)} images saved. Metadata stored in '{OUTPUT_JSON}'.")
