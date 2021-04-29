from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pyramid import pyramid


def main():
    img = np.array(Image.open("img2.jpeg").convert("RGB")) / 255.0
    _, _, _, p = pyramid(img)

    f, ax = plt.subplots(len(p) + 1, figsize=(8, 20))
    ax[0].imshow(img)

    ax[0].imshow(img)
    ax[0].axis("off")

    for i in range(len(p)):
        ax[i+1].imshow(p[i])
        ax[i+1].axis("off")

    plt.show()


if __name__ == '__main__':
    main()
