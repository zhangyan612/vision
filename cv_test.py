import cv2
print(cv2.__version__)
# import sys
# print(sys.path)
sift = cv2.xfeatures2d.SIFT_create()
detector = cv2.SIFT()