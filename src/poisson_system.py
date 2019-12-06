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

    def __init__(self, source, destination, mask, offset=[0, 0]):
        self.g = Poisson_system._affine_transform(source, destination.shape[:2], offset=offset).astype(float)
        self.f = destination.astype(float)
        self.mask = mask
        self.map = Map(mask)

        # create kernel
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.Np = correlate2d(np.ones(mask.shape), self.kernel, mode='same')
        self.A = self._get_A()
        self.method_list = [["dg", "import_gradients", "basic seamless cloning"],
                            ["dgf", "mixing_gradients", "transparent seamless cloning"],
                            ["Mdg", "masked_gradients", "texture flattening"],
                            ["abf", "nonlinear transformed gradients", "local illumination changes"]]
        self.cur_method = self.b = None

    def get_Ab(self, method='dg', **kwargs):
        """
        return A and b from the discrete of the guidance field v under different methods.
        :param method: choose different method to get different guidance field v.
            - "dg"/ "import gradients"/ "basic seamless cloning": v = \partial g (default)
            - "dgf"/ "mixing gradients"/ "transparent seamless cloning": v = max(\partial g, \partial f)
            - "Mdf"/ "masked gradients"/ "texture flattening": v = M \partial f
            - "abf"/ "nonlinear transformed gradients"/ "local illumination changes": v=a^b|\partial f|^{-b}\partial f
        :return: the discrete of the guidance field v under different methods.
        """
        if (self.cur_method is None) or (method not in self.cur_method):
            if method in self.method_list[0]:
                v = self._import_gradients(**kwargs)
            elif method in self.method_list[1]:
                v = self._mixing_gradients(**kwargs)
            elif method in self.method_list[2]:
                v = self._masked_gradients(**kwargs)
            else:
                raise ValueError(
                    "Not defined guidance vector methods. Please select from {}".format(self.print_methods()))
            self.b = self._get_b(v)
        # already calculated
        assert (self.A is not None) and (self.b is not None)
        return self.A, self.b

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

    def _mixing_gradients(self):
        v = np.zeros(self.g.shape)
        place = ['[:-1, :]', '[1:, :]', '[:, :-1]', '[:, 1:]']
        idx = [(0, 1), (2, 3), (1, 0), (3, 2)]
        for i in range(4):
            g, f = self.g.copy(), self.f.copy()
            exec('g' + place[idx[i][0]] + ' -= g' + place[idx[i][1]])
            exec('f' + place[idx[i][0]] + ' -= f' + place[idx[i][1]])
            v += np.where(abs(f) > abs(g), f, g)
        return v[self.mask == 1]

    def _masked_gradients(self):
        edge = cv2.Canny(cv2.cvtColor(self.f.astype(np.uint8), cv2.COLOR_RGB2GRAY), 100, 300)
        v = np.zeros(self.g.shape)
        place = [':-1, :', '1:, :', ':, :-1', ':, 1:']
        idx = [(0, 1), (2, 3), (1, 0), (3, 2)]
        for i in range(4):
            exec('f = self.f[' + place[idx[i][0]] + '] - self.f[' + place[idx[i][1]] + ']')
            exec('v[' + place[idx[i][0]] +
                 '] += np.where(edge[' + place[idx[i][0]] + ', np.newaxis] != edge[' +
                 place[idx[i][1]] + ', np.newaxis], f, 0)')
        return v[self.mask == 1]

    def combine(self, x):
        res = self.f.copy()
        res[self.mask == 1] = x
        res[res > 255] = 255
        res[res < 0] = 0

        return res / 255
