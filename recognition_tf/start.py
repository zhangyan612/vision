import os

cmd = (
    "Python object_detection_app.py -src {url} {option}").format(
        url ='http://192.168.0.47:8080/video?.mjpeg',
        option='') # -num-w 5
os.system(cmd)
