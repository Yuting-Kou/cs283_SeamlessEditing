import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.signal import correlate2d

class Map:
    def __init__(self, mask):
        self.mask = mask.astype(int)
        self.bnd = self.mask & \
                   cv2.dilate(1-mask, np.array([[0,1,0], [1,0,1], [0,1,0]], dtype = np.uint8)).astype(int)
        outbnd = (~self.mask) & \
                 cv2.dilate(mask, np.array([[0,1,0], [1,0,1], [0,1,0]], dtype = np.uint8)).astype(int)
        tmp = np.nonzero(outbnd)
        self.outbnd = set(zip(tmp[0], tmp[1]))
        tmp = np.nonzero(mask)
        self.idx2pos = dict(zip(list(zip(list(tmp[0]), list(tmp[1]))), range(len(tmp[0]))))

    def in_omega(self, pos):
        return self.mask[pos] == 1

    def is_boundary(self, pos):
        return self.bnd[pos] == 1


class Poisson_system:

    @staticmethod
    def _affine_transform(Iin, output_size, offset):
        """
        transform image (source img) into the output_size(destination) with offset
        :param Iin: source img
        :param output_size: output size
        :param offset=[tx,ty], where output = [x+tx, y+ty]
        """
        M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        return cv2.warpAffine(Iin, M, (output_size[1], output_size[0]))

    def __init__(self, source, destination, mask, offset = [0,0]):
        self.g = Poisson_system._affine_transform(source, destination.shape[:2], offset=offset).astype(float)
        self.f = destination.astype(float)
        self.mask = mask
        self.map = Map(mask)
        self.Np = correlate2d(np.ones(mask.shape), np.array([[0,1,0], [1,0,1], [0,1,0]]), mode = 'same')

        self.A = self._get_A()

    def get_Ab(self, method = 'dg'):
        if method == 'dg':
            v = self._laplacian()
            b = self._get_b(v)
        return self.A, b

    def _get_A(self):
        n = (self.mask == 1).sum()
        A = lil_matrix((n, n))
        for i, pos in enumerate(self.map.idx2pos.keys()):
            A[i, i] = self.Np[pos]
            for dx, dy in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                new_x, new_y = pos[0] + dx, pos[1] + dy
                if (new_x, new_y) in self.map.idx2pos:
                    A[i, self.map.idx2pos[(new_x, new_y)]] = -1
        return A

    def _get_b(self, v):
        b = v
        idx = np.nonzero(self.map.bnd)
        for x, y in zip(idx[0], idx[1]):
            for dx, dy in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                new_x, new_y = x + dx, y+dy
                if (new_x, new_y) in self.map.outbnd:
                    b[self.map.idx2pos[(x, y)]] += self.f[new_x, new_y]
        return b

    def _laplacian(self):
        g = np.zeros(self.g.shape)
        for i in range(3):
            g[:,:,i] = self.Np * self.g[:,:,i]- \
                correlate2d(self.g[:,:,i], np.array([[0,1,0], [1,0,1], [0,1,0]]), mode='same')
        return g[self.mask == 1]

    def combine(self, x):
        x[x > 255] = 255
        x[x < 0] = 0

        res = self.f.copy()
        res[self.mask == 1] = x

        return res/255








