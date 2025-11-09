# generate_dataset.py
import os, random, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

def draw_char_image(ch, img_size=(28,28), font_size=20):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Compatibilidad con Pillow >= 10
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Pillow < 10
        w, h = draw.textsize(ch, font=font)

    pos = ((img_size[0] - w) // 2, (img_size[1] - h) // 2)
    draw.text(pos, ch, fill=0, font=font)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img


def create_dataset(root='data_chars'):
    chars = list('0123456789' + string.ascii_uppercase)
    for subset in ['train','test']:
        for c in chars:
            os.makedirs(os.path.join(root, subset, c), exist_ok=True)
    for c in chars:
        for i in range(50):
            draw_char_image(c).save(f'{root}/train/{c}/{c}_{i}.png')
        for i in range(10):
            draw_char_image(c).save(f'{root}/test/{c}/{c}_{i}.png')
    print('Dataset generado en', root)

if __name__ == '__main__':
    create_dataset()
