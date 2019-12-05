import matplotlib.pyplot as plt
from src.gradient_field import Gradient_Field
import numpy as np
if __name__ == '__main__':


    destination = plt.imread(r'../test/swan.jpg')
    source = plt.imread(r'../test/kyt.jpg')
    print(source.shape, destination.shape)

    mask = plt.imread(r'../test/mask.png')
    mask[mask != 1] = 0
    mask = mask.astype(np.uint8)[:, :, 0]

    test = Gradient_Field(source=source, destination=destination, mask=mask, offset=[-136, 50])
    A, b = test.get_v()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(test.f_star)
    # plt.subplot(1, 2, 2)
    # plt.imshow(test.g)
    # plt.show()