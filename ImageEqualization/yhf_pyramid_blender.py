import cv2
import numpy as np

A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

class PyramidBlender():
    def __init__(self, i1, i2, mask, layers):
        self._i1 = i1
        self._i2 = i2
        self._mask = (mask==0).astype(np.uint8)
        self._n = layers
        self.blendImage = None

    def _generatePyramid(self):
        n = self._n
        # generate Gaussian pyramids
        G = self._i1.copy()
        self._gp1 = [G]
        h,w,c = G.shape
        self._sizep = [(w,h)]
        for i in range(n-1):
            G = cv2.pyrDown(G)
            h, w, c = G.shape
            self._sizep.append((w, h))
            self._gp1.append(G)

        G = self._i2.copy()
        self._gp2 = [G]
        for i in range(n-1):
            G = cv2.pyrDown(G)
            self._gp2.append(G)

        # Generate mask pyramid
        G = self._mask.copy()
        self._maskp = [G]
        for i in range(n-1):
            G = (cv2.pyrDown(G) > 0.5).astype('uint8')
            self._maskp.append(G)

        # generate Laplacian Pyramids
        # pay attention to the uint8 substraction
        self._lp1 = []
        for i in range(n-1):
            GE = cv2.pyrUp(self._gp1[i+1], dstsize = self._sizep[i])
            L = self._gp1[i].astype(np.int) - GE
            self._lp1.append(L)
        self._lp1.append(self._gp1[-1])

        self._lp2 = []
        for i in range(n - 1):
            GE = cv2.pyrUp(self._gp2[i + 1], dstsize=self._sizep[i])
            L = self._gp2[i].astype(np.int) - GE
            self._lp2.append(L)
        self._lp2.append(self._gp2[-1])

    def _combine(self):
        combine_lp = []
        for i in range(self._n):
            mask1 = self._maskp[i]
            mask2 = np.subtract(1, mask1)
            temp = np.multiply(self._lp1[i], mask1).astype(np.int) + np.multiply(self._lp2[i], mask2).astype(np.int)
            combine_lp.append(temp)

        combine_total = combine_lp[-1]
        combine_total = self._convertToUint8(combine_total)
        for i in range(self._n-1, 0, -1):
            combine_total = cv2.pyrUp(combine_total, dstsize = self._sizep[i-1])
            combine_total = cv2.add(combine_total.astype(np.int), combine_lp[i-1])
            combine_total = self._convertToUint8(combine_total)
        self.blendImage = combine_total

    def _convertToUint8(self, a):
        a[a > 255] = 255
        a[a < 0] = 0
        return a.astype(np.uint8)

    def run(self):
        self._generatePyramid()
        self._combine()
        return self.blendImage

if __name__ == "__main__":
    i1 = cv2.imread('./data/green_cup.jpg')
    i2 = cv2.imread('./data/blue_cup.jpg')
    mask1 = cv2.imread('./data/cup_mask.bmp')
    imageBlender = PyramidBlender(i1, i2, mask1, 6)
    output_image = imageBlender.run()
    cv2.imwrite("./yhf_blend.jpg", output_image)