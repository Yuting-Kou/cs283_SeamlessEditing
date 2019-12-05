import matplotlib.pyplot as plt
import numpy as np

from src.gradient_field import Gradient_Field
from src.mask import MaskPainter

if __name__ == '__main__':
    paths = '../test/'
    destination = 'swan.jpg'
    source = 'kyt.jpg'
    save = 'mask.png'

    mp = MaskPainter(g_impath=paths + source, f_impath=paths + destination)
    tx, ty, savepath = mp.paint_mask(maskname=save)
    print('save mask in' + savepath)

    mask = plt.imread(paths + save)
    mask[mask != 1] = 0
    mask = mask.astype(np.uint8)[:, :, 0]  # only 1 channel mask

    test = Gradient_Field(source=mp.image_g, destination=mp.image_f, mask=mask, offset=[tx, ty],
                          neighbor_ker=np.ones((5, 5)))

    print('Choose methods from:', test.print_methods())

    A, b = test.get_v(method='dgf')

    print(A.shape, b.shape)
    # do not need to calculate the same thing
    A1, b1 = test.get_v(method='dgf')
    # calculate new methods.
    A2, b2 = test.get_v(method='dg')
    A3, b3 = test.get_v(method='Mdg', new_mask=np.ones_like(test.g))