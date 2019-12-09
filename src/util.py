import cv2
import numpy as np


def affine_transform(Iin, output_size, offset=(0, 0)):
    """
    transform image (source img) into the output_size(destination) with offset
    :param Iin: source img
    :param output_size: output size
    :param offset=[tx,ty], where output = [x+tx, y+ty]
    """
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    return cv2.warpAffine(Iin.copy(), M, (output_size[1], output_size[0]))

def resize_fix_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    """ resize with fixed ratio"""
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    elif width is None:
        dim = (int(w * height * 1.0 / h), height)
    else:
        dim = (width, int(h * width * 1.0 / w))
    return cv2.resize(img, dim, interpolation=inter)

def blend(image, mask, x, alpha=0.3):
    ret = image.copy()
    ret[mask != 0] = ret[mask != 0] * alpha + x[mask != 0] * (1 - alpha)
    return ret.astype(np.uint8)