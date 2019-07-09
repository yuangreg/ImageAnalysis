import cv2
import numpy as np

class ImageMatcher:
	def __init__(self):
		self._featureDetector = cv2.xfeatures2d.SURF_create()
		# self._featureDetector = cv2.xfeatures2d.SIFT_create()
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
		# self._match = cv2.BFMatcher()

	def _getFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self._featureDetector.detectAndCompute(gray, None)
		return kp, des

	def match(self, i1, i2):
		"""
		:param i1: original image plane
		:param i2: target image plane
		:return: transformation matrix
		"""
		kp1, des1 = self._getFeatures(i1)
		kp2, des2 = self._getFeatures(i2)
		matches = self._matcher.knnMatch(des1, des2, k=2)

		good = []
		for m, n in matches:
			if m.distance < 0.2 * n.distance:
				good.append(m)

		if 0:
			# Draw point correspondence
			draw_params = dict(matchColor=(0, 255, 0),
						   singlePointColor=None,
						   flags=2)
			img_match = cv2.drawMatches(i1, kp1, i2, kp2, good, None, **draw_params)
			cv2.imwrite("yhf.png", img_match)

		if len(good) > 4:
			src = np.float32(
				[kp1[m.queryIdx].pt for m in good]
			)
			dest = np.float32(
				[kp2[m.trainIdx].pt for m in good]
				)

			H, s = cv2.findHomography(src, dest, cv2.RANSAC, 4)
			return H
		return None


if __name__ == "__main__":
	i1 = cv2.imread('./carmel/carmel-01.png')
	i2 = cv2.imread('./carmel/carmel-00.png')
	matcher = ImageMatcher()
	H = matcher.match(i1, i2)