import cv2
import numpy as np
from scipy.signal import correlate2d
from scipy.sparse import lil_matrix


class Map:
    def __init__(self, mask):
        self.mask = mask.astype(int)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.bnd = self.mask & \
                   cv2.dilate(1 - mask, kernel).astype(int)
        outbnd = (~self.mask) & \
                 cv2.dilate(mask, kernel).astype(int)
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

    @staticmethod
    def create_kernel(neighbor_ker=4):
        if isinstance(neighbor_ker, int):
            assert (neighbor_ker == 4) or (neighbor_ker) == 8
            if neighbor_ker == 4:
                neigh_ker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
            else:
                neigh_ker = np.ones((3, 3), dtype=np.uint8)
        else:
            # allow larger neighbor range.
            assert neighbor_ker.shape[0] == neighbor_ker.shape[1] and neighbor_ker.shape[0] % 2 == 1
            neigh_ker = np.array(neighbor_ker, dtype=np.uint8)
        mid = neigh_ker.shape[0] // 2
        neigh_ker[mid, mid] = 0  # not include itself.
        return neigh_ker

    def __init__(self, source, destination, mask, offset=[0, 0], ker=4):
        self.g = Poisson_system._affine_transform(source, destination.shape[:2], offset=offset).astype(float)
        self.f = destination.astype(float)
        self.mask = mask
        self.map = Map(mask)

        # create kernel
        self.kernel = Poisson_system.create_kernel(neighbor_ker=ker)
        self.Np = correlate2d(np.ones(mask.shape), self.kernel, mode='same')
        self.A = self._get_A()
        self.method_list = [["dg", "import gradients", "basic seamless cloning"],
                            ["dgf", "mixing gradients", "transparent seamless cloning"],
                            ["Mdg", "masked gradients", "texture flattening"],
                            ["abf", "nonlinear transformed gradients", "local illumination changes"]]
        self.cur_method = self.b = None

    def get_Ab(self, method='dg'):
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
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) in self.map.outbnd:
                    b[self.map.idx2pos[(x, y)]] += self.f[new_x, new_y]
        return b

    def _laplacian(self):
        g = np.zeros(self.g.shape)
        for i in range(3):
            g[:, :, i] = self.Np * self.g[:, :, i] - \
                         correlate2d(self.g[:, :, i], self.kernel, mode='same')
        return g[self.mask == 1]

    def combine(self, x):
        res = self.f.copy()
        res[self.mask == 1] = x
        res[res > 255] = 255
        res[res < 0] = 0

        return res / 255