from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pyramid import pyramid


def main():
    img = np.array(Image.open("test_image.png").convert("RGB")) / 255.0
    f, ax = plt.subplots(6, figsize=(8, 20))
    ax[0].imshow(img)
    _, p, _, rr = pyramid(img)

    ax[0].imshow(img)
    ax[0].axis("off")

    for i in range(3):
        ax[i+1].imshow(p[i])
        ax[i+1].axis("off")

    plt.show()


if __name__ == '__main__':
    main()
