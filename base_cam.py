import cv2

class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(0)
        if not self.cam:
            raise Exception("Camera not accessible")

        self.shape = self.get_frame().shape


    def get_frame(self):
        _, frame = self.cam.read()
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ =='__main__':
    print('start camera')
    # Camera().run()
