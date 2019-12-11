import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from scipy.sparse import lil_matrix

from src.util import affine_transform


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
    method_list = [["dg", "import gradients", "basic seamless cloning"],
                   ["dgf", "mixing gradients", "transparent seamless cloning"],
                   ["Mdg", "masked gradients", "texture flattening"],
                   ["ilm", "illumination", "change local ilumination"]]

    @staticmethod
    def regu_mask(mask):
        """make sure mask is 2D and only 0 and 1"""
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(float)
        mask[mask != 0] = 1
        return mask

    def __init__(self, source, destination, mask, offset=[0, 0], reshape=False, adjust_ilu=False, alpha=0.5):
        self.g = affine_transform(source, destination.shape[:2], offset=offset).astype(float) if not reshape \
            else source.astype(float)
        self.f = destination.astype(float)
        self.f_reset = self.f.copy()
        self.mask = Poisson_system.regu_mask(mask)
        self.map = Map(self.mask)

        # create kernel
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.Np = correlate2d(np.ones(self.mask.shape), self.kernel, mode='same')

        # balance illuminance
        if adjust_ilu:
            self.balance_illuminance(alpha=alpha)

        self.A = self._get_A()
        self.cur_method = self.b = None

    def balance_illuminance(self, alpha=0.5):
        """Make the source image to have similar illuminance as destination area. """
        near_bnd = cv2.dilate(self.map.bnd.astype(np.uint8), kernel=self.kernel, iterations=1)
        ilu_diff = self.g[self.mask == 1].mean() - self.f[near_bnd == 1].mean()
        self.f[near_bnd == 1] += alpha * ilu_diff

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
            if method in Poisson_system.method_list[0]:
                self.cur_method = Poisson_system.method_list[0]
                print('Import gradient', self.cur_method)
                v = self._laplacian(self.g)
            elif method in Poisson_system.method_list[1]:
                self.cur_method = Poisson_system.method_list[1]
                print('mixing gradient', self.cur_method)
                v = self._mixing_gradients(**kwargs)
            elif method in Poisson_system.method_list[2]:
                self.cur_method = Poisson_system.method_list[2]
                print('masked gradient', self.cur_method)
                v = self._masked_gradients(**kwargs)
            elif method in Poisson_system.method_list[3]:
                self.cur_method = Poisson_system.method_list[3]
                print('illumination gradient', self.cur_method)
                v = self._illumination_gradients(**kwargs)
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

    def _laplacian(self, x):
        v = np.zeros(x.shape)
        for i in range(3):
            v[:, :, i] = self.Np * x[:, :, i] - \
                         correlate2d(x[:, :, i], self.kernel, mode='same')
        return v[self.mask == 1]

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

    def _masked_gradients(self, low=100, high=300):
        edge = cv2.Canny(cv2.cvtColor(self.g.astype(np.uint8), cv2.COLOR_RGB2GRAY), 100, 300)
        v = np.zeros(self.g.shape)
        place = [':-1, :', '1:, :', ':, :-1', ':, 1:']
        idx = [(0, 1), (2, 3), (1, 0), (3, 2)]
        for i in range(4):
            exec('g = self.g[' + place[idx[i][0]] + '] - self.g[' + place[idx[i][1]] + ']')
            exec('v[' + place[idx[i][0]] +
                 '] += np.where(edge[' + place[idx[i][0]] + ', np.newaxis] != edge[' +
                 place[idx[i][1]] + ', np.newaxis], g, 0)')
        return v[self.mask == 1]

    def _illumination_gradients(self):
        df = self._laplacian(self.g)
        beta = 0.2
        v = np.zeros(df.shape)
        for i in range(3):
            blur = gaussian_filter(self.g[:, :, i], sigma=3)
            dx = correlate2d(blur, np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), mode='same')
            dy = correlate2d(blur, np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]), mode='same')
            norm = (dx ** 2 + dy ** 2) ** 0.5
            alpha = 0.2 * norm[self.mask == 1].mean()
            v[:, i] = (alpha / norm[self.mask == 1]) ** beta * df[:, i]
        return v

    def combine(self, x):
        res = self.f_reset.copy()
        res[self.mask == 1] = x
        res[res > 255] = 255
        res[res < 0] = 0

        return res / 255
