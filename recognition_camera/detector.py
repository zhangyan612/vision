import cv2
import os
import numpy as np
from PIL import Image 
import pickle


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


recognizer = cv2.face.LBPHFaceRecognizer_create() # createLBPHFaceRecognizer()
recognizer.read('trainer/trainer.yml')

cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataSet'

cam = cv2.VideoCapture() # 0
cam.open("http://192.168.0.47:8080/video?.mjpeg")

#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) #Creates a font
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
        print(nbr_predicted)
        if nbr_predicted == 1:
             nbr_predicted = 'Yan Zhang'
        elif nbr_predicted == 2:
             nbr_predicted='Diana Barron'
        # cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        #cv2.PutText(cv2.fromarray(im), str(nbr_predicted)+"--"+str(conf), (x, y+h), font, 255) # Draw the text
        draw_text(im, str(nbr_predicted)+"--"+ str(conf), x, y+h)
        #cv2.putText(im, str(nbr_predicted)+"--"+str(conf), (x, y+h), cv2.FONT_HERSHEY_PLAIN, 255)  # Draw the text

        cv2.imshow('im', im)
        cv2.waitKey(10)









