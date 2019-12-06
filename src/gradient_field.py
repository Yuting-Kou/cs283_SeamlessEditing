import cv2
import numpy as np
import pandas as pd
from scipy.signal import correlate2d
from scipy.sparse import diags, coo_matrix


class Gradient_Field:
    """
    Set different guidance field v to achieve different functional tools of seamless editing.
    This object has method of get_vector with different method.

    Notes: when add a new vector method, remember to update self.method_list in __init__ and get_v function.
    """

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

    def _laplacian(self, In):
        """ take gradient of In image."""
        if len(In.shape) == 3:
            res = np.zeros_like(In)
            for i in range(3):
                res[:, :, i] = self.Np * In[:, :, i] - correlate2d(In[:, :, i], self.neigh_ker, mode='same')
        else:
            res = self.Np * In - correlate2d(In, self.neigh_ker, mode='same')
        return res

    def boundary_smooth(self):
        """b1 terms: sum up the neighbor value of f at boundary"""
        if len(self.f_star.shape) == 3:
            b1 = np.zeros_like(self.f_star)
            for i in range(3):
                b1[:, :, i] = correlate2d(self.f_star[:, :, i] * self.boundary, self.neigh_ker, mode='same')
        else:
            b1 = correlate2d(self.f_star * self.boundary, self.neigh_ker, mode='same')
        return b1

    def __init__(self, source, destination, mask, offset=[0, 0], neighbor_ker=4):
        """
        create a gradient field instance
        :param source: source image g which is the pixel function within the selected area.
            Notes: the source image g will be affine transformed into same size of f* (destination img).
        :param offset: set the position of offset area when conduct above affine transformation.
                offset[0]:tx, move rightwards. offset[1]: ty. move downwards.
        :param destination: destination image f* which is the pixel function outside of the selected area.
        :param mask: corresponds to the region of source image that will be copied and
        be placed into the desired location in destination img. It has same shapes as destination image f*.
        :param neighbor_ker: int or squared array. choose 4 or 8 neighbors, or user_default boarder kernel.
        """

        self.g = Gradient_Field._affine_transform(source, destination.shape[:2], offset=offset)
        self.f_star = destination
        self.Ns = int(mask.sum())
        self.mask = mask.astype(bool)  # only True or False
        assert self.mask.shape == self.f_star.shape[:2]

        # create neighbor kernel
        if isinstance(neighbor_ker, int):
            assert (neighbor_ker == 4) or (neighbor_ker) == 8
            if neighbor_ker == 4:
                self.neigh_ker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
            else:
                self.neigh_ker = np.ones((3, 3), dtype=np.uint8)
        else:
            # allow larger neighbor range.
            assert neighbor_ker.shape[0] == neighbor_ker.shape[1] and neighbor_ker.shape[0] % 2 == 1
            self.neigh_ker = np.array(neighbor_ker, dtype=np.uint8)
        mid = self.neigh_ker.shape[0] // 2
        self.neigh_ker[mid, mid] = 0  # not include itself.

        self.Np = correlate2d(np.ones(self.mask.shape, dtype=np.uint8), self.neigh_ker, mode='same')  # 2d
        self.boundary = cv2.dilate(src=self.mask.astype(np.uint8), kernel=self.neigh_ker) - self.mask.astype(np.uint8)
        self.method_list = [["dg", "import gradients", "basic seamless cloning"],
                            ["dgf", "mixing gradients", "transparent seamless cloning"],
                            ["Mdg", "masked gradients", "texture flattening"],
                            ["abf", "nonlinear transformed gradients", "local illumination changes"]]
        self.cur_method = self.b = None

        self.A = self._calA()
        self.b1 = self.boundary_smooth()

    def print_methods(self):
        """helper function, print methods"""
        return self.method_list

    def _calA(self):
        # calculate A: same value for all the channels. Use row-wise flatten.
        a2 = self.mask.ravel()
        m, n = self.g.shape[:2]
        klists = []
        K = self.neigh_ker.shape[0]
        mid = K // 2
        for i in range(K):
            for j in range(K):
                klists.append((i - mid) * n + (j - mid))

        # A is still the full matrix. but outside the region A is Identity matrix
        A = diags(self.Np.flatten())
        for k in np.array(klists)[self.neigh_ker.ravel() == 1]:
            val = np.zeros(n * m)
            if k > 0:
                z = np.r_[a2[k:], np.zeros(k)]
                val[z == 1] = -1
                A.setdiag(val, k=k)
            else:
                z = np.r_[np.zeros(-k), a2[:k]]
                val[z == 1] = -1
                A.setdiag(val[-k:], k=k)
        # only inside region
        A = A.tocoo()
        idx = np.where(self.mask.ravel())[0]
        tmp = pd.DataFrame({'row': A.row, 'col': A.col, 'data': A.data})
        tmp.index = tmp['row']
        tmp = tmp.loc[idx]
        tmp.index = tmp['col']
        tmp = tmp.loc[idx]
        tmp['new_row'] = tmp.row.rank(method='dense') - 1
        tmp['new_col'] = tmp.col.rank(method='dense') - 1
        return coo_matrix((tmp.data, (tmp.new_row.astype(int), tmp.new_col.astype(int))), shape=(self.Ns, self.Ns))

    def get_v(self, method="dg", **kwargs):
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
                self._import_gradients(**kwargs)
            elif method in self.method_list[1]:
                self._mixing_gradients(**kwargs)
            elif method in self.method_list[2]:
                self._masked_gradients(**kwargs)
            elif method in self.method_list[3]:
                self._nonlinear_transformed_gradients(**kwargs)
            else:
                raise ValueError(
                    "Not defined guidance vector methods. Please select from {}".format(self.print_methods()))
        # already calculated
        assert (self.A is not None) and (self.b is not None)
        return self.A, self.b

    def _import_gradients(self):
        """ v=grad g"""
        self.cur_method = self.method_list[0]
        print('Importing gradients: v=dg')
        self.b = (self.b1 + self._laplacian(self.g))[self.mask]

    def _mixing_gradients(self):
        """mixing gradient: transparent importing."""
        self.cur_method = self.method_list[1]
        print('Mixing gradient: v= max(dg, df_star)')
        grad_g = self._laplacian(self.g)
        grad_f = self._laplacian(self.f)
        new_grad = np.where(grad_f > grad_g, grad_f, grad_g)
        self.b = (self.b1 + new_grad)[self.mask]

    def _masked_gradients(self, new_mask, g_eq_f=True):
        """
        The gradient is only passed through a sparse sieve that retains only the most salient features
        If g = f_star, it can realize in-place image transformations.
        :param new_mask: Binary mask that turned on at a few locations of interests. E.g. edge detector.
        """
        self.cur_method = self.method_list[2]
        print('Salient mask gradient: v=M*dg')
        target = self.f_star if g_eq_f else self.g
        assert new_mask.shape[:2] == target.shape[:2]
        if len(new_mask.shape) > len(target.shape):
            new_mask = new_mask[:, :, 0]
        new_mask = new_mask.astype(bool)
        grad = self._laplacian(target)
        if len(grad.shape) > len(new_mask):
            self.b = (self.b1 + grad * new_mask[:, :, np.newaxis])[self.mask]
        else:
            self.b = (self.b1 + grad * new_mask)

    def _nonlinear_transformed_gradients(self):
        self.cur_method = self.method_list[3]
        pass
