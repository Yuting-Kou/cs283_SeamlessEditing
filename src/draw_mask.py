"""
reference: https://github.com/PPPW/poisson-image-editing/blob/master/paint_mask.py
Use mouse to draw mask
"""

import argparse
from os import path

import cv2
import numpy as np


class MaskPainter():
    def __init__(self, impath):
        """size is the width in pixel of your drawer"""
        self.image = cv2.imread(impath)
        assert self.image is not None
        self.image_path = impath
        self.mask = np.zeros_like(self.image)
        self.draw = False
        self.size = 5
        self.image_rest = self.image.copy()
        self.mask_reset = self.mask.copy()
        self.window_name = "Draw mask:   s-save; r:reset; q:quit; +:larger painter; -:smaller painter"

    def _paint(self, event, x, y, flags, param):
        # click and draw
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.rectangle(self.image, (x - self.size, y - self.size), (x + self.size, y + self.size),
                              (0, 255, 0), -1)
                cv2.rectangle(self.mask, (x - self.size, y - self.size), (x + self.size, y + self.size),
                              (255, 255, 255), -1)
                cv2.imshow(self.window_name, self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

    def paint_mask(self, maskname='mask.png'):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._paint)

        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            if (key == ord("r")) or (key == ord("R")):
                self.image = self.image_rest.copy()
                self.mask = self.mask_reset.copy()

            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()
            elif key == ord("+"):
                self.size += 1
            elif key == ord("-"):
                self.size -= 1

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        if '.' not in maskname:
            maskname = maskname + '.png'
        maskpath = path.join(path.dirname(self.image_path), maskname)
        cv2.imwrite(maskpath, self.mask)

        # close all open windows
        cv2.destroyAllWindows()
        return maskpath


if __name__ == '__main__':

    source_path = r'../test/kyt.jpg'

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-s", "--save", required=False, help="mask name to save")
    args = vars(ap.parse_args())

    mp = MaskPainter(args["image"])
    if "save" in args:
        print('save mask in ', mp.paint_mask(maskname=args["save"]))
