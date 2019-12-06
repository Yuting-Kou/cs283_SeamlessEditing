"""
reference: https://github.com/PPPW/poisson-image-editing/blob/master/paint_mask.py
Use mouse to draw mask
"""

import argparse
from os import path

import cv2
import numpy as np


class MaskPainter():
    def __init__(self, g_impath, f_impath):
        """size is the width in pixel of your drawer"""
        self.image_g = cv2.imread(g_impath)
        assert self.image_g is not None
        if f_impath is None:
            self.image_f = self.image_g
        else:
            self.image_f = cv2.imread(f_impath)
            assert self.image_f is not None
        self.f_path = f_impath
        self.g_path = g_impath
        self.mask = np.zeros_like(self.image_g)
        self.draw = False
        self.size = 5
        self.image_g_reset = self.image_g.copy()
        self.image_f_reset = self.image_f.copy()
        self.mask_reset = self.mask.copy()
        self.original_mask_copy = np.zeros(self.image_f.shape)
        self.window_name = "Draw mask:   s-save; r:reset; q:quit; l:larger painter; m:smaller painter"
        self.window_name_move = "Move mask:   s-save; r:reset; q:quit;"
        self.to_move = False
        self.x0 = 0
        self.y0 = 0
        self.is_first = True
        self.xi = 0
        self.yi = 0

    def _paint(self, event, x, y, flags, param):
        # click and draw
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.rectangle(self.image_g, (x - self.size, y - self.size), (x + self.size, y + self.size),
                              (0, 255, 0), -1)
                cv2.rectangle(self.mask, (x - self.size, y - self.size), (x + self.size, y + self.size),
                              (255, 255, 255), -1)
                cv2.imshow(self.window_name, self.image_g)
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

    def _blend(self, image, mask):
        ret = image.copy()
        alpha = 0.3
        ret[mask != 0] = ret[mask != 0] * alpha + 255 * (1 - alpha)
        return ret.astype(np.uint8)

    def _move_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_move = True
            if self.is_first:
                self.x0, self.y0 = x, y
                self.is_first = False

            self.xi, self.yi = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_move:
                M = np.float32([[1, 0, x - self.xi],
                                [0, 1, y - self.yi]])
                self.mask = cv2.warpAffine(self.mask, M,
                                           (self.mask.shape[1],
                                            self.mask.shape[0]))
                cv2.imshow(self.window_name,
                           self._blend(self.image_f, self.mask))
                self.xi, self.yi = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.to_move = False

    def paint_mask(self, maskname='mask.png'):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._paint)
        cv2.resizeWindow(self.window_name, 600, 600)

        while True:
            cv2.imshow(self.window_name, self.image_g)
            key = cv2.waitKey(1) & 0xFF

            if (key == ord("l")) or (key == ord("L")):  # up or plus
                self.size += 1
                # print("larger", self.size)
            elif (key == "m") or (key == ord("M")):
                self.size -= 1
                # print("smaller", self.size)
            elif (key == ord("r")) or (key == ord("R")):
                self.image_g = self.image_g_reset.copy()
                self.mask = self.mask_reset.copy()

            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)

        # shift mask
        self.original_mask_copy[np.where(self.mask != 0)] = 255

        self.mask = self.original_mask_copy.copy()
        cv2.namedWindow(self.window_name_move)
        cv2.setMouseCallback(self.window_name_move,
                             self._move_mask_handler)

        while True:
            cv2.imshow(self.window_name_move,
                       self._blend(self.image_f, self.mask))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image_f = self.image_f_reset.copy()
                self.mask = self.original_mask_copy.copy()

            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        if '.' not in maskname:
            maskname = maskname + '.png'
        new_mask_path = path.join(path.dirname(self.f_path),
                                  maskname)
        cv2.imwrite(new_mask_path, self.mask)

        # close all open windows
        cv2.destroyAllWindows()
        # close all open windows
        cv2.destroyAllWindows()
        return self.xi - self.x0, self.yi - self.y0, new_mask_path


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--source_image", required=True, help="Path to the source image")
    ap.add_argument("-f", "--destination_image", required=False,
                    help="Path to the destination image. Default is same as source")
    ap.add_argument("-s", "--save", required=False, help="mask name to save")
    args = vars(ap.parse_args())

    mp = MaskPainter(g_impath=args["source_image"], f_impath=args["destination_image"])
    if args["save"] is not None:
        print('save mask in {2}, offset=({0},{1})'.format( mp.paint_mask(maskname=args["save"])))
    else:
        print('save mask in  {2}, offset=({0},{1})'.format( mp.paint_mask()))
