from os import path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from src.util import affine_transform, resize_fix_ratio, blend


class Painter:
    def __init__(self, g_impath, f_impath=None):
        self.g = cv2.imread(g_impath)
        self.f = cv2.imread(f_impath) if f_impath is not None else self.g.copy()
        self.g_name = g_impath.split('/')[-1]
        self.path = f_impath if f_impath is not None else g_impath

        # draw params
        self.size = 5
        self.mask = np.zeros(self.g.shape[:2])
        self.g_reset = self.g.copy()
        self.mask_reset = self.mask.copy()
        self.draw = False
        self.window_name_draw = "Draw mask:  s-save; r:reset; q:quit; l:large painter; m:small painter, p: change mode"
        self.ix, self.iy = -1, -1
        self.mode = True  # if True, draw rectangle. Press 'p' to toggle to curve

        # shift params
        self.resize = False
        self.window_name_move = "Move mask:  s-save; r:reset; q:quit; p: change mode"
        self.to_move = False
        self.xi, self.yi, = 0, 0

    def get_mask(self, maskname='mask.png'):
        self.draw_mask()
        return self.shift_mask(maskname=maskname)

    def _draw_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                if self.mode:
                    # draw ractangle
                    cv2.rectangle(self.g, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
                    cv2.rectangle(self.mask, (self.ix, self.iy), (x, y), (255, 255, 255), -1)
                else:
                    # draw line
                    cv2.rectangle(self.g, (x - self.size, y - self.size), (x + self.size, y + self.size),
                                  (0, 255, 0), -1)
                    cv2.rectangle(self.mask, (x - self.size, y - self.size), (x + self.size, y + self.size),
                                  (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

    def draw_mask(self):
        """
        allow draw mask in the source image.
        Function: P: change mode to draw rectangle or draw curve points
                  L/M: make painter larger or smaller under mode=True (draw curve mode)
                  R: reset mask
                  S: save mask
                  Q: quit and unsaved
        """
        cv2.namedWindow(self.window_name_draw, 0)
        cv2.setMouseCallback(self.window_name_draw, self._draw_mask_handler)

        while 1:
            cv2.imshow(self.window_name_draw, self.g)
            k = cv2.waitKey(1) & 0xFF

            if (k == ord('p')) or (k == ord('P')):
                self.mode = not self.mode
            elif (k == ord("l")) or (k == ord("L")):  # up or plus
                self.size += 1
                # print("larger", self.size)
            elif (k == "m") or (k == ord("M")):
                self.size -= 1
                # print("smaller", self.size)
            elif (k == ord("r")) or (k == ord("R")):
                self.g = self.g_reset.copy()
                self.mask = self.mask_reset.copy()
            elif k == ord("s"):
                break
            elif (k == ord("q")) or (k == ord("Q")):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _move_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_move = True
            self.xi, self.yi = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_move:
                self.mask = affine_transform(self.mask, self.mask.shape[:2], (x - self.xi, y - self.yi))
                self.g = affine_transform(self.g, self.mask.shape[:2], (x - self.xi, y - self.yi))
                cv2.imshow(self.window_name_move,
                           blend(self.f, self.mask, self.g))
                self.xi, self.yi = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.to_move = False

    def _change(self):
        # get value from trackbar
        value = cv2.getTrackbarPos("resize", self.window_name_move)
        print('resize', round(value / 10, 2))
        h, w = self.mask.shape[:2]
        # resize image
        width = int(w * 1.0 * value / 10)
        self.mask = resize_fix_ratio(self.mask, width=width)
        self.mask = affine_transform(self.mask, self.f.shape[:2])
        self.g = resize_fix_ratio(self.g, width=width)
        self.g = affine_transform(self.g, self.f.shape[:2])

        cv2.setTrackbarPos("resize", self.window_name_move, 10)

    def shift_mask(self, maskname='mask.png'):
        """
        Allow shift the mask under the destination image
        Function: tracebar to adjust size of mask after press P
                  R: reset mask
                  S: save mask
                  Q: quit and unsaved
        maskname: name to store final mask.
        """
        self.mask_reset = np.zeros(self.f.shape[:2])
        # resize image if source is larger than destination
        h, w = self.f.shape[:2]
        ptsidx = np.where(self.mask != 0)
        fxy = min(h * 1.0 / max(ptsidx[0]), w * 1.0 / max(ptsidx[1]))
        if fxy < 1:
            print('resize')
            width = int(w * fxy)
            self.mask = resize_fix_ratio(self.mask, width=width)
            self.g = resize_fix_ratio(self.g_reset, width=width)
            self.g = affine_transform(self.g, self.f.shape)
            self.g_reset = self.g.copy()
            ptsidx = np.where(self.mask != 0)
        else:
            self.g = affine_transform(self.g_reset, self.f.shape)
        self.mask_reset[ptsidx] = 255
        self.mask = self.mask_reset.copy()

        cv2.namedWindow(self.window_name_move, 0)
        cv2.setMouseCallback(self.window_name_move,
                             self._move_mask_handler)
        # create trackbar
        cv2.createTrackbar("resize", self.window_name_move, 10, 20, self._change)

        while True:
            cv2.imshow(self.window_name_move,
                       blend(self.f, self.mask, self.g))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r") or key == ord("R"):
                self.mask = self.mask_reset.copy()
                self.g = self.g_reset.copy()
            elif key == ord("s") or key == ord("S"):
                break
            elif key == ord("p") or key == ord("P"):
                self._change()
            elif key == ord("q") or key == ord("Q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        self.mask /= self.mask.max()
        self.mask[self.mask != 0] = 1.0
        if '.' not in maskname:
            maskname = maskname + '.png'
        new_mask_path = path.join(path.dirname(self.path),
                                  maskname)
        cv2.imwrite(new_mask_path, self.mask)
        new_g_path = path.join(path.dirname(self.path),
                               'new_' + self.g_name)
        cv2.imwrite(new_g_path, self.g)

        # close all open windows
        cv2.destroyAllWindows()
