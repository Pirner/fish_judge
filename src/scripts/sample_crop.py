import cv2


def main():
    img_path = r"C:\data\fish_judge\lama\im.png"
    mask_path = r"C:\data\fish_judge\lama\mask.png"

    im = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    xmin = 24
    ymin = 50
    xmax = 250
    ymax = 350
    im_crop = im[ymin:ymax, xmin:xmax]
    mask_crop = mask[ymin:ymax, xmin:xmax]

    cv2.imwrite(r"C:\data\fish_judge\lama\im_crop.png", im_crop)
    cv2.imwrite(r"C:\data\fish_judge\lama\mask_crop.png", mask_crop)


if __name__ == '__main__':
    main()
