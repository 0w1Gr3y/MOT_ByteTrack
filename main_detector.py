from VideoPlayer import VideoPlayer
import cv2 as cv
from Predictor import Predictor
import argparse
import time
from utils import show_image

# python .\main_detector.py --name yolox-m --ckpt weights/yolox_m.pth --video_input assets/KarolMajek720.avi --video_output output_yolox-det-m.avi
if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX + ByteTrack Demo")
    # video input and outputs
    parser.add_argument("--video_input", type=str, default='assets/video.avi', required=True,
                        help="path to input video file")
    parser.add_argument("--video_output", type=str, default='video_output.avi', required=True,
                        help="path to output video file")

    # YOLOX arguments
    parser.add_argument("-n", "--name", default='yolox-m', type=str, required=True,
                        help="model name [yolox-s, yolox-m, yolox-l, yolox-x]")
    parser.add_argument("-c", "--ckpt", default='weights/yolox_m.pth', type=str, required=True,
                        help="weights file for the model")
    parser.add_argument("--det_thresh", type=float, default=0.1, help="YOLOX confidence threshold")
    parser.add_argument("--det_nmsthresh", type=float, default=0.4,
                        help="YOLOX non-maximum supression threshold threshold")
    args = parser.parse_args()

    predictor = Predictor(ckpt=args.ckpt, name=args.name, confthre=args.det_thresh, nmsthre=args.det_nmsthresh)
    video_input_path = args.video_input
    video_output_path = args.video_output

    camera_player = VideoPlayer(video_input_path)
    if camera_player.video_cap is None:
        exit(-1)

    video_writer = cv.VideoWriter(video_output_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), camera_player.fps,
                                  (camera_player.width, camera_player.height))
    while True:
        ret, frame = camera_player.get_frame()
        if ret is False:
            break

        outputs, img_info = predictor.inference(frame)
        time_detector = predictor.det_time
        vis_res = predictor.visual(outputs, img_info)
        if vis_res.shape[1] > 1280:
            show_image('frame', vis_res, (1280, -1))
        else:
            cv.imshow('frame', vis_res)
        video_writer.write(vis_res)

        cv_key = cv.waitKey(1)
        if cv_key is ord('q'):
            break
    video_writer.release()
    cv.destroyAllWindows()
    print("Finished")
