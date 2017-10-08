import numpy as np
import cv2
import os

cap = cv2.VideoCapture()
# Opening the link
cap.open("http://192.168.0.47:8080/video?.mjpeg")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('Mobile IP Camera', frame)
    # Clear screen
    os.system('clear')
    # Exit key
    print("Press 'q' to exit")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()