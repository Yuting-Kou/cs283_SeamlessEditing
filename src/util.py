import cv2
import numpy as np
from IPython.display import HTML


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
    ret = ret*1.0/ret.max()
    x = x.copy()
    x = x*1.0/x.max()
    ret[mask != 0] = ret[mask != 0] * alpha + x[mask != 0] * (1 - alpha)
    return ret.astype(float)


def hidecode():
    return HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    The raw code for this IPython notebook is by default hidden for easier reading.
    To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

