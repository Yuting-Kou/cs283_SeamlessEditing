import time

import matplotlib.pyplot as plt
import numpy as np

from src.equation import Poisson_system
from src.gradient_field import Gradient_Field

if __name__ == '__main__':
    paths = '../test/'
    destination = 'swan.jpg'
    source = 'kyt.jpg'
    save = 'mask.png'

    # mp = MaskPainter(g_impath=paths + source, f_impath=paths + destination)
    # tx, ty, savepath = mp.paint_mask(maskname=save)
    # print('save mask in' + savepath)
    # print('offset=({0},{1})'.format(tx,ty))
    source = plt.imread(source)  # source=mp.image_g
    destination = plt.imread(destination)  # destination=mp.image_f

    tx, ty = (-184, -158)
    mask = plt.imread(paths + save)
    mask[mask != 1] = 0
    mask = mask.astype(np.uint8)[:, :, 0]  # only 1 channel mask

    begin = time.time()
    test = Gradient_Field(source=source, destination=destination, mask=mask, offset=[tx, ty],  neighbor_ker=4)
    A, b = test.get_v(method='dg')
    print('It costs {:.2f} sec.'.format(time.time() - begin))
    begin = time.time()
    test1 = Poisson_system(source=source, destination=destination, mask=mask, offset=[tx, ty])
    A1, b1 = test1.get_Ab(method='dg')
    print('It costs {:.2f} sec.'.format(time.time() - begin))

    # print('Choose methods from:', test.print_methods())
    #

    # x = sor_solver(A, b[0], 1.5, np.zeros(A.shape[0]), 1e-6)

    #
    # print(A.shape, b.shape)
    # # do not need to calculate the same thing
    # # A1, b1 = test.get_v(method='dgf')
    # # # calculate new methods.
    # # A2, b2 = test.get_v(method='dg')
    # # A3, b3 = test.get_v(method='Mdg', new_mask=np.ones_like(test.g))
    #
    # g=test.g.copy()
    # g[test.boundary == 0]=1
    # plt.imshow(g)
    # plt.show()
    # plt.imshow(test.g)
    # plt.show()
