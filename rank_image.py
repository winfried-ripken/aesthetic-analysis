from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pyramid import pyramid


def main():
    img = np.array(Image.open("test_image.png").convert("RGB")) / 255.0
    lps, ds, dhats, fs = pyramid(img)

    f, ax = plt.subplots(len(fs) + 1, figsize=(8, 20))

    ax[0].imshow(img)
    ax[0].axis("off")

    for i in range(len(fs)):
        ax[i+1].imshow(fs[i])
        ax[i+1].axis("off")

    plt.show()

    contrast_image = np.zeros_like(ds[0])
    for d in ds:
        contrast_image += d

    binary_fs = [f > 0.0 for f in fs]
    in_focus_map = binary_fs[0]
    out_of_focus_map = ~in_focus_map
    out_of_focus_area = np.mean(out_of_focus_map)

    sharpness = np.mean(fs[0])
    depth = np.argmax(np.array([np.sum(f) for f in binary_fs[1:]])) + 2
    clarity = out_of_focus_area * (np.mean(contrast_image * in_focus_map) - np.mean(contrast_image * out_of_focus_map))

    # TODO color & tone metric

    print("sharpness", sharpness)
    print("depth", depth)
    print("clarity", clarity)


if __name__ == '__main__':
    main()
