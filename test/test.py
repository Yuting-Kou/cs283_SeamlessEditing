import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.get_mask import Painter
from src.poisson_solver import sor_solver_jit
from src.poisson_system import Poisson_system
from src.util import blend

if __name__ == '__main__':
    paths = '../test/'
    destination = 'f.jpg'
    source = 'g.jpg'
    save = 'mask.png'
    draw = True

    if draw:
        pp = Painter(g_impath=paths + source, f_impath=paths + destination)
        pp.get_mask()
        source = cv2.cvtColor(pp.g, cv2.COLOR_BGR2RGB)
        destination = cv2.cvtColor(pp.f, cv2.COLOR_BGR2RGB)
        mask=pp.mask.astype(np.uint8)

        test = Poisson_system(source=source, destination=destination, mask=pp.mask.astype(np.uint8))
    else:
        source = plt.imread('new_'+source)  # source=mp.image_g
        destination = plt.imread(destination)  # destination=mp.image_f

        mask = plt.imread(paths + save)
        mask[mask != 1] = 0
        mask = mask.astype(np.uint8)  # only 1 channel mask
        test = Poisson_system(source=source, destination=destination, mask=mask)

    sourcetmp = blend(source, mask, x=np.ones_like(source) * 255)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sourcetmp)
    plt.title('source')
    plt.subplot(1, 2, 2)
    plt.imshow(destination)
    plt.title('destination')
    plt.show()
    begin = time.time()
    # test = Gradient_Field(source=source, destination=destination, mask=mask, offset=[tx, ty], ker=4)

    A, b = test.get_Ab(method='dgf')
    A = A.tolil()

    data = [np.array(it) for it in A.data]
    rows = [np.array(it) for it in A.rows]
    diag = A.diagonal()
    omega = 1.8
    x0 = np.zeros(b.shape)
    x1 = sor_solver_jit(data, rows, diag, b, omega, x0)
    plt.imshow(test.combine(x1))
    plt.show()
    # print('It costs {:.2f} sec.'.format(time.time() - begin))
    # begin = time.time()
    # test1 = Poisson_system(source=source, destination=destination, mask=mask, offset=[tx, ty])
    # A1, b1 = test1.get_Ab(method='dg')
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
