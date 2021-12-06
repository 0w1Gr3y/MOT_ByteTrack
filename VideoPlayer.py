import cv2 as cv


class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_cap = cv.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            print("Error opening video stream or file")
            self.video_cap = None
            return
        self.fps = int(self.video_cap.get(cv.CAP_PROP_FPS))
        self.cv_wait_time = int(1000 / self.fps)
        self.width = int(self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print('inited video capture \"{}\" | FPS: {} | Resolution: {}x{}'.format(video_path, self.fps,
                                                                                 self.video_cap.get(
                                                                                     cv.CAP_PROP_FRAME_WIDTH),
                                                                                 self.video_cap.get(
                                                                                     cv.CAP_PROP_FRAME_HEIGHT)))

    def get_frame(self):
        ret, frame = self.video_cap.read()
        return ret, frame

    def reset(self):
        self.video_cap.release()
        self.video_cap = cv.VideoCapture(self.video_path)

    def __del__(self):
        if self.video_cap is not None:
            self.video_cap.release()
