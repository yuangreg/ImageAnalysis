import cv2
import numpy as np

class GetPolygon:
    def __init__(self, im):
        self.im = im
        self.points = []
        self.points_array = []
        self._start = 0

    def drawpolygon(self, event, x, y, flags, param):
        if(self._start==0):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._start = 1
                self.points.append((x, y))
                self.points_array.append([x, y])
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                pt1 = self.points[-1]
                pt2 = (x, y)
                cv2.line(self.im,pt1,pt2,(0,0,255),1)
                self.points.append(pt2)
                self.points_array.append([x, y])

    def run(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.drawpolygon)

        while (1):
            cv2.imshow('image', self.im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
        cv2.destroyAllWindows()
        return self.points_array

if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)
    getpoly = GetPolygon(img)
    points = getpoly.run()
    print(points)

