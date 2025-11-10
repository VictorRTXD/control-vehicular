# generate_plates.py
import os, random, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

OUT = "data/images"
os.makedirs(OUT, exist_ok=True)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

def random_plate():
    L = lambda: random.choice(LETTERS)
    D = lambda: random.choice(DIGITS)
    return f"{L()}{D()}{D()}-{L()}{L()}{L()}"

def make_A00_examples(n=200):
    # genera variaciones del caso A00-AAA
    examples = []
    for i in range(n):
        plate = "A00-" + ''.join(random.choices(LETTERS, k=3))
        examples.append(plate)
    return examples

def random_font(size):
    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for f in fonts:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default()

def apply_random_effects(img):
    # small rotation, blur and gaussian noise
    img = img.rotate(random.uniform(-6,6), resample=Image.BICUBIC, fillcolor=(255,255,255))
    if random.random() < 0.6:
        img = ImageOps.autocontrast(img)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0,1)))
    arr = np.array(img).astype(np.float32)
    if random.random() < 0.7:
        arr += np.random.normal(0, random.uniform(0,18), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def create_plate_image(plate_text, size=(128,64)):
    bg = Image.new('RGB', size, (255,255,255))
    draw = ImageDraw.Draw(bg)
    font = random_font(int(size[1]*0.6))
    bbox = draw.textbbox((0,0), plate_text, font=font)
    w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    draw.text(((size[0]-w)/2, (size[1]-h)/2), plate_text, fill=(0,0,0), font=font)
    return apply_random_effects(bg)

def generate(n=4000, outdir=OUT):
    os.makedirs(outdir, exist_ok=True)
    # include specific A00-AAA examples to improve recognition of that pattern
    a00_list = make_A00_examples(400)
    i = 0
    # first add A00 examples
    for plate in a00_list:
        img = create_plate_image(plate)
        img.save(os.path.join(outdir, f"{i}_{plate}.png"))
        i += 1
    # then general plates
    while i < n:
        plate = random_plate()
        img = create_plate_image(plate)
        img.save(os.path.join(outdir, f"{i}_{plate}.png"))
        i += 1
    print(f"✅ Generadas {n} imágenes en {outdir}")

if __name__ == "__main__":
    generate(4000)
