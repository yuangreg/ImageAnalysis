import cv2
import numpy as np
import matplotlib.pyplot as plt

class InteractiveMask:
    def __init__(self, im):
        self.im = im.copy()
        self.mask = 128*np.ones(im.shape[:2],np.uint8)
        self._draw = False
        self._mode = 'F'

    def getMask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._draw = True
            self._mode = 'F'
            cv2.circle(self.im, (x, y), 5, (0, 255, 0), thickness = 5)
            cv2.circle(self.mask, (x, y), 5, 255, thickness = 5)

        elif event == cv2.EVENT_LBUTTONUP:
            self._draw = False

        if event == cv2.EVENT_MOUSEMOVE:
            if self._draw == True:
                if self._mode == 'F':
                    cv2.circle(self.im, (x, y), 5, (0, 255, 0), thickness=5)
                    cv2.circle(self.mask, (x, y), 5, 255, thickness=5)
                else:
                    cv2.circle(self.im, (x, y), 5, (0, 0, 255), thickness=5)
                    cv2.circle(self.mask, (x, y), 5, 0, thickness=5)

        if event == cv2.EVENT_RBUTTONDOWN:
            self._draw = True
            self._mode = 'B'
            cv2.circle(self.im, (x, y), 5, (0, 0, 255), thickness = 5)
            cv2.circle(self.mask, (x, y), 5, 0, thickness = 5)

        elif event == cv2.EVENT_RBUTTONUP:
            self._draw = False

    def run(self):
        cv2.namedWindow('Please Label Image', cv2.WINDOW_NORMAL)
        h, w, c = self.im.shape
        cv2.resizeWindow('Please Label Image', int(480/h*w), 480)
        cv2.setMouseCallback('Please Label Image', self.getMask)

        while (1):
            cv2.imshow('Please Label Image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
        cv2.destroyAllWindows()
        return self.mask

if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)
    getMask = InteractiveMask(img)
    mask = getMask.run()
    plt.imshow(mask, cmap='gray')
    plt.show()


