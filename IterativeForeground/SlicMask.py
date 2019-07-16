import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries

class SlicMask:
    def __init__(self, im):
        self.im = im.copy()
        self.im_copy = self.im.copy()
        self.mask = np.zeros(im.shape[:2],np.uint8)
        self._draw = False
        self._mode = "F"
        self._numSegments = 500
        self._segments = None

    def imageSlic(self):
        self._segments = slic(self.im, n_segments=self._numSegments, sigma=0)
        self._segments_copy = self._segments.copy()
        plt.imshow(mark_boundaries(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB), self._segments))
        plt.show()

    def getMask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if(self._segments[y, x] != -1):
                self._draw = True
                self._mode = "F"
                self._segments[self._segments == self._segments[y, x]] = -1
                self.im[self._segments == self._segments[y, x], :] = (0, 255, 0)

        if event == cv2.EVENT_MOUSEMOVE:
            if self._draw == True:
                if self._mode == "F":
                    self._segments[self._segments == self._segments[y, x]] = -1
                    self.im[self._segments == self._segments[y, x], :] = (0, 255, 0)
                else:
                    self._segments[self._segments_copy == self._segments_copy[y, x]] = self._segments_copy[y, x]
                    self.im[self._segments_copy == self._segments_copy[y, x], :] = self.im_copy[self._segments_copy == self._segments_copy[y, x], :]

        elif event == cv2.EVENT_LBUTTONUP:
            self._draw = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self._draw = True
            self._mode = 'B'
            self._segments[self._segments_copy == self._segments_copy[y, x]] = self._segments_copy[y, x]
            self.im[self._segments_copy == self._segments_copy[y, x], :] = self.im_copy[self._segments_copy == self._segments_copy[y, x], :]

        elif event == cv2.EVENT_RBUTTONUP:
            self._draw = False

    def run(self):
        self.imageSlic()
        cv2.namedWindow('Please Label Image')
        cv2.setMouseCallback('Please Label Image', self.getMask)

        while (1):
            cv2.imshow('Please Label Image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
        cv2.destroyAllWindows()
        self.mask[self._segments == -1] = 255
        return self.mask

if __name__ == "__main__":
    img = cv2.imread("./data/cat.jpg")
    getMask = SlicMask(img)
    mask = getMask.run()
    object = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(cv2.cvtColor(object, cv2.COLOR_BGR2RGB))
    plt.show()


