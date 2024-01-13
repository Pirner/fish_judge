from simple_lama_inpainting import SimpleLama
from PIL import Image


def main():
    simple_lama = SimpleLama()

    img_path = r"C:\data\fish_judge\lama\im_crop.png"
    mask_path = r"C:\data\fish_judge\lama\mask_crop.png"

    image = Image.open(img_path)
    mask = Image.open(mask_path).convert('L')

    result = simple_lama(image, mask)
    result.save("inpainted.png")


if __name__ == '__main__':
    main()
