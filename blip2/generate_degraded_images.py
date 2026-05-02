import os
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import argparse
import io

parser = argparse.ArgumentParser()
parser.add_argument("--level", choices=["medium", "heavy"], required=True)
args = parser.parse_args()

INPUT_DIR = "../data/img"
OUTPUT_DIR = f"../data/img_{args.level}"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def jpeg_compress(img, quality=30):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def degrade_image(img, level):
    w, h = img.size

    if level == "medium":
        img = img.resize((int(w * 0.5), int(h * 0.5)), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        return img

    if level == "heavy":
        img = img.resize((int(w * 0.25), int(h * 0.25)), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img = ImageEnhance.Contrast(img).enhance(0.8)
        img = jpeg_compress(img, quality=30)

        return img

    return img


for filename in tqdm(os.listdir(INPUT_DIR)):
    if not filename.endswith(".png"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        img = Image.open(input_path).convert("RGB")
        img = degrade_image(img, args.level)
        img.save(output_path)
    except Exception as e:
        print(f"Failed on {filename}: {e}")
