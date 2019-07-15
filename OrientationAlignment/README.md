# Orientation Alignment

### Automatically detect the four corners of a book/letter.
- Smooth the image
```
gray = cv2.medianBlur(gray, 21)
```
* Obtain all the lines
```
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
```
* Delete similar lines
```
self.lines = self._deleteRepeat(lines)
```
* Find the intersection and order them w.r.t. their centroid
```
self._sortPoints()
```

### Apply perspective transformation to convert to front view.