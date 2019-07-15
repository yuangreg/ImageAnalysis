import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class BoundaryDetection:
    def __init__(self, im):
        self.im = im
        self.lines = []
        self.points = []

    def findCornerPoints(self):
        h, w, c = self.im.shape
        if len(self.lines)==4:
            for i in range(4):
                for j in range(i+1, 4):
                    line1 = self.lines[i]
                    line2 = self.lines[j]
                    [x, y] = self._findIntersection(line1, line2)
                    if(x>=0 and x<w and y>=0 and y<h):
                        self.points.append([x, y])
        else:
            return
        if 0:
            im = self.im.copy()
            for point in self.points:
                x, y = point
                cv2.circle(im, (x, y), 10, (0, 255, 0), 2)
            plt.imshow(im)
            plt.show()

        self._sortPoints()

    def _sortPoints(self):
        # Sort Anticlockwise
        points = np.array(self.points)
        sum_x = np.sum(points[:, 0])
        sum_y = np.sum(points[:, 1])
        centroid = [sum_x / 4, sum_y / 4]

        sortedPoints = sorted(points, key=lambda coord: math.atan2(coord[0]-centroid[0], coord[1]-centroid[1]))
        self.points = list(sortedPoints)

    def _findIntersection(self, line1, line2):
        """Finds the intersection of two lines
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]


    def findLines(self):
        # Turn to grayscale and filter the image, e.g., Gaussian filter, median filter, bilateral filter
        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.bilateralFilter(gray, 100, sigmaColor = 10, sigmaSpace = 500)
        gray = cv2.medianBlur(gray, 21)

        if 0:
            plt.imshow(gray, cmap='gray')
            plt.show()

        # Use Canny and Hough transform to find boundaries
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        self.lines = self._deleteRepeat(lines)


    def _deleteRepeat(self, lines):
        rho_gap = 100
        theta_gap = 45 * np.pi / 180
        output_line_set = []

        for line in lines:
            rho, theta = line[0]
            # Search for similar lines
            skip = False
            for exist_line in output_line_set:
                if abs(exist_line[0]-rho)<rho_gap and abs(exist_line[1]-theta)<theta_gap:
                    skip = True
            if skip:
                continue
            else:
                output_line_set.append((rho, theta))

        if 0:
            im = self.im.copy()
            for rho, theta in output_line_set:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plt.imshow(im)
            plt.show()

        return output_line_set

    def run(self):
        self.findLines()
        self.findCornerPoints()
        return self.points

if __name__ == "__main__":
    img = cv2.imread("./data/photo-large.jpeg")
    bd = BoundaryDetection(img)
    points = bd.run()