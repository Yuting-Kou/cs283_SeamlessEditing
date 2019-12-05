import cv2
import numpy as np
from scipy.signal import correlate2d
from scipy.sparse import diags


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

    def __init__(self, source, destination, mask, offset=[0, 0], neighbor_ker=4):
        """
        create a gradient field instance
        :param source: source image g which is the pixel function within the selected area.
            Notes: the source image g will be affine transformed into same size of f* (destination img).
        :param offset: set the position of offset area when conduct above affine transformation.
        :param destination: destination image f* which is the pixel function outside of the selected area.
        :param mask: corresponds to the region of source image that will be copied and
        be placed into the desired location in destination img. It has same shapes as destination image f*.
        :param neighbor_ker: int or squared array. choose 4 or 8 neighbors, or user_default boarder kernel.
        """

        self.g = Gradient_Field._affine_transform(source, destination.shape[:2], offset=offset)
        self.f_star = destination
        self.mask = mask.astype(np.uint8)
        print('mask value set:',np.unique(self.mask))

        # neighbor
        if isinstance(neighbor_ker, int):
            assert (neighbor_ker == 4) or (neighbor_ker) == 8
            if neighbor_ker == 4:
                self.neighbor_ker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
            else:
                self.neighbor_ker = np.ones((3, 3), dtype=np.uint8)
        else:
            assert neighbor_ker.shape == (3, 3)
            self.neighbor_ker = np.array(neighbor_ker, dtype=np.uint8)
        self.neighbor_ker[1, 1] = 0  # not include itself.
        self.Np = correlate2d(np.ones(self.mask.shape, dtype=np.uint8), self.neighbor_ker, mode='same')  # 2d
        self.boundary = cv2.dilate(src=self.mask, kernel=self.neighbor_ker) - self.mask
        self.method_list = [["dg", "import gradients", "basic seamless cloning"],
                            ["dgf", "mixing gradients", "transparent seamless cloning"],
                            ["Mdf", "masked gradients", "texture flattening"],
                            ["abf", "nonlinear transformed gradients", "local illumination changes"]]
        self.cur_method = self.A = self.b = None

        # calculate b1: sum of f_star(q) where q is at boundary;
        if len(self.g.shape) == 3:
            self.b1 = [
                correlate2d(self.f_star[:, :, i] * self.boundary, self.neighbor_ker, mode='same')
                for i in range(self.f_star.shape[2])]
        else:
            self.b1 = correlate2d(self.f_star * self.boundary, self.neighbor_ker, mode='same')

        # calculate A: same value for all the channels. Use row-wise flatten.
        a2 = self.mask.flatten()
        m, n = self.g.shape[:2]
        klists = np.array([-n - 1, -n, -n + 1, -1, 0, 1, n - 1, n, n + 1])

        self.A = diags(self.Np.flatten())
        for k in klists[self.neighbor_ker.flatten() == 1]:
            val = np.zeros(int(m * n))
            if k > 0:
                z = np.r_[a2[k:], np.zeros(k)]
                val[z == 1] = -1
                self.A.setdiag(val, k=k)
            else:
                z = np.r_[np.zeros(-k), a2[:k]]
                val[z == 1] = -1
                self.A.setdiag(val[-k:], k=k)

    def print_methods(self):
        """helper function, print methods"""
        return self.method_list

    def get_v(self, method="dg"):
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
                self._import_gradients()
            elif method in self.method_list[1]:
                self._mixing_gradients()
            elif method in self.method_list[2]:
                self._masked_gradients()
            elif method in self.method_list[3]:
                self._nonlinear_transformed_gradients()
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
        # b2 sum of gradient vector: g_p - g_q. q is neighbor
        if len(self.g.shape) == 3:
            self.b = np.array([(self.b1[i] + self.Np * self.g[:, :, i] - correlate2d(self.g[:, :, i], self.neighbor_ker,
                                                                   mode='same')).flatten()
                           for i in range(self.g.shape[2])])

        else:
            b2 = self.Np * self.g - correlate2d(self.g, self.neighbor_ker, mode='same')
            self.b = (self.b+b2).flatten()


    def _mixing_gradients(self):
        self.cur_method = self.method_list[1]
        pass

    def _masked_gradients(self):
        self.cur_method = self.method_list[2]
        pass

    def _nonlinear_transformed_gradients(self):
        self.cur_method = self.method_list[3]
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    source = plt.imread(r'../test/swan.jpg')
    destination = plt.imread(r'../test/kyt.jpg')
    print(source.shape, destination.shape)

    ker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    mask = np.zeros(destination.shape[:2], np.uint8)
    mask[500:-500, 500:-500] = 1
    test = Gradient_Field(source=source, destination=destination, mask=mask)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test.g)
    plt.subplot(1, 2, 2)
    plt.imshow(test.f_star)
    plt.show()
    A,b = test.get_v()
