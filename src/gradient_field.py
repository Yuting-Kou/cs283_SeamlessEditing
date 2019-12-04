import cv2
import numpy as np
from scipy.signal import correlate2d


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
        return cv2.warpAffine(source, M, (output_size[0], output_size[1]))

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
        self.mask = mask

        # neighbor
        if len(neighbor_ker) < 2:
            assert (neighbor_ker == 4) or (neighbor_ker) == 8
            if neighbor_ker == 4:
                self.neighbor_ker = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            else:
                self.neighbor_ker = np.ones((3, 3))
        else:
            assert neighbor_ker.shape == (3, 3)
            self.neighbor_ker = neighbor_ker
        self.neighbor_ker[1, 1] = 0  # not include itself.
        self.Np = correlate2d(np.ones(self.mask.shape), self.neighbor_ker, mode='same')  # 2d
        self.boundary = cv2.dilate(src=self.mask, kernel=self.neighbor_ker) - test.mask
        self.method_list = [["dg", "import gradients", "basic seamless cloning"],
                            ["dgf", "mixing gradients", "transparent seamless cloning"],
                            ["Mdf", "masked gradients", "texture flattening"],
                            ["abf", "nonlinear transformed gradients", "local illumination changes"]]
        self.cur_method = self.A = self.b = None

        # calculate b1: sum of f_star(q) where q is at boundary;
        if len(self.g.shape) == 3:
            self.b1 = correlate2d(self.f_star * self.boundary[:, :, np.newaxis], self.neighbor_ker, mode='same')
        else:
            self.b1 = correlate2d(self.f_star * self.boundary, self.neighbor_ker, mode='same')

        # calculate A


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
        return self.A, self.b

    def _import_gradients(self):
        """ v=grad g"""
        self.cur_method = self.method_list[0]
        # b2 sum of gradient vector: g_p - g_q. q is neighbor
        if len(self.g.shape) == 3:
            b2 = self.Np[:, :, np.newaxis] * self.g - correlate2d(self.g, self.neighbor_ker, mode='same')
        else:
            b2 = self.Np * self.g - correlate2d(self.g, self.neighbor_ker, mode='same')
        self.b = self.b1 + b2


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

    test = Gradient_Field(source=source, destination=destination, mask=np.zeros(destination.shape[:2]))
    test.get_v()
