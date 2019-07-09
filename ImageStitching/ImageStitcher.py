import cv2
import numpy as np
from ImageMatcher import ImageMatcher

class ImageStitcher:
    # warp i1 to i2
    def __init__(self):
        self._i1 = None
        self._i2 = None
        self._pre_shift = None
        self._post_shift = None
        self.mergeImage = None
        self.mask1_out = None
        self.mask2_out = None
        self._imageMatcher = ImageMatcher()

    def initialize(self, i1, i2, mask1=None, mask2=None):
        """

        :param i1: original image plane
        :param i2: target image plane
        :param mask1: [x_min, x_max, y_min, y_max] for i1
        :param mask2: [x_min, x_max, y_min, y_max] for i2
        :return: merged image, 4 coordinate of i1 in merged image, 4 coordinate of i2 in merged image
        """

        self._i1 = i1
        self._i2 = i2
        self._mask1 = mask1
        self._mask2 = mask2

        if mask1 is None:
            self._pre_shift = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            self._pre_shift = [[1, 0, -mask1[2]], [0, 1, -mask1[0]], [0, 0, 1]]
        if mask2 is None:
            self._post_shift = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            self._post_shift = [[1, 0, mask2[2]], [0, 1, mask2[0]], [0, 0, 1]]

    def merge(self):
        if(self._mask1 is None):
            i1_patch = self._i1
        else:
            i1_patch = self._i1[self._mask1[0]:self._mask1[1], self._mask1[2]:self._mask1[3], :]
        if(self._mask2 is None):
            i2_patch = self._i2
        else:
            i2_patch = self._i2[self._mask2[0]:self._mask2[1], self._mask2[2]:self._mask2[3], :]
        M = self._imageMatcher.match(i1_patch, i2_patch)

        if(M is not None):
            condNumb = self._checkConditionNumber(M)
            print("Conditional Number is ", condNumb)
            if(condNumb > 20):
                print("Conditional Number is too large")
                return None, None, None
        else:
            print("No match found")
            return None, None, None


        M_with_preshift = np.matmul(M, self._pre_shift)
        h, w, c = self._i1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M_with_preshift)
        offset_x = int(max(np.ceil((dst[:, :, 0]).min() * -1), 0))
        offset_y = int(max(np.ceil((dst[:, :, 1]).min() * -1), 0))
        M_negtive_offset = [[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]]
        overall_offset = np.matmul(self._post_shift, M_negtive_offset)
        overall_M = np.matmul(overall_offset, M_with_preshift)

        patch_dst = cv2.perspectiveTransform(pts, overall_M)
        patch_x_max = int(np.ceil((patch_dst[:, :, 0]).max()))
        patch_x_min = max(int(np.floor((patch_dst[:, :, 0]).min())), 0)
        patch_y_max = int(np.ceil((patch_dst[:, :, 1]).max()))
        patch_y_min = max(int(np.floor((patch_dst[:, :, 1]).min())), 0)

        overall_x_size = max(offset_x + self._i2.shape[1], patch_x_max)
        overall_y_size = max(offset_y + self._i2.shape[0], patch_y_max)
        self.mergeImage = cv2.warpPerspective(self._i1, overall_M, (overall_x_size, overall_y_size))
        self.mergeImage[offset_y:offset_y + self._i2.shape[0], offset_x:offset_x + self._i2.shape[1]] = self._i2

        self.mask1_out = [patch_y_min, patch_y_max, patch_x_min, patch_x_max]
        self.mask2_out = [offset_y ,offset_y + self._i2.shape[0], offset_x , offset_x + self._i2.shape[1]]

        self._trim()
        if 0:
            cv2.imwrite("./yhf1.png", self.mergeImage)

        return self.mergeImage, self.mask1_out, self.mask2_out



    def _checkConditionNumber(self, H):
        shearMatrix = H[0:2, 0:2]
        condNum = np.linalg.cond(shearMatrix)
        return condNum

    def _trim(self):
        frame = self.mergeImage
        h, w, c = frame.shape
        #crop top
        for i in range(0,h):
            if(np.sum(frame[i])==0):
                continue
            else:
                top = i
                break

        for i in range(h-1, 0, -1):
            if (np.sum(frame[i]) == 0):
                continue
            else:
                bottom = i + 1
                break

        for j in range(0, w):
            if (np.sum(frame[:,j])==0):
                continue
            else:
                left = j
                break

        for j in range(w-1, 0, -1):
            if (np.sum(frame[:,j])==0):
                continue
            else:
                right = j + 1
                break

        self.mergeImage = frame[top:bottom, left:right, :]
        self.mask1_out = np.add(self.mask1_out , [-top, -top, -left, -left])
        self.mask2_out = np.add(self.mask2_out , [-top, -top, -left, -left])


if __name__ == "__main__":
    i1 = cv2.imread('./carmel/carmel-00.png')
    i2 = cv2.imread('./carmel/carmel-01.png')
    i3 = cv2.imread('./carmel/carmel-02.png')
    imageStitcher = ImageStitcher()
    imageStitcher.initialize(i1, i2)
    mergeImaged, mask1, mask2 = imageStitcher.merge()

    imageStitcher.initialize(mergeImaged, i3, mask1 = mask1)
    mergeImaged, mask1, mask2 = imageStitcher.merge()
