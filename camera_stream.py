import cv2

class CameraStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # mirror view for user-facing camera

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
