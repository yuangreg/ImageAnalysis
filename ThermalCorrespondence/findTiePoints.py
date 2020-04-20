import cv2
import numpy as np

class GetTiePoints:
    def __init__(self, im):
        self.im = im
        self.imcopy = im.copy()
        self.pts = []

    def getPoints(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
            cv2.circle(self.im, (x, y), 30, (0, 0, 255), 10)
            cv2.putText(self.im, str(len(self.pts)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 10)
        else:
            if event == cv2.EVENT_RBUTTONDOWN:
                self.pts = []
                self.im = self.imcopy.copy()

    def run(self):
        cv2.namedWindow('Please Label Two Tie Points', cv2.WINDOW_NORMAL)
        h, w, c = self.im.shape
        cv2.resizeWindow('Please Label Two Tie Points', int(480 / h * w), 480)
        cv2.setMouseCallback('Please Label Two Tie Points', self.getPoints)

        while (1):
            cv2.imshow('Please Label Two Tie Points', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
        cv2.destroyAllWindows()
        return np.array(self.pts)

if __name__ == "__main__":
    img_rgb = cv2.imread("./images/rgb.jpg")
    getPoints = GetTiePoints(img_rgb)
    points = getPoints.run()
    print(points)

