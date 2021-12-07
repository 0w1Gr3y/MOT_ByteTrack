from VideoPlayer import VideoPlayer
import cv2 as cv
from Predictor import Predictor
from tracker.byte_tracker import BYTETracker
from tracker.utils import plot_tracking
import argparse
import time
from utils import show_image

# python .\main.py --name yolox-m --ckpt weights/yolox_m.pth --video_input assets/KarolMajek720.avi --video_output output_yolox-m.avi
if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX + ByteTrack Demo")
    # video input and outputs
    parser.add_argument("--video_input", type=str, default='assets/video.avi', required=True,
                        help="path to input video file")
    parser.add_argument("--video_output", type=str, default='video_output.avi', required=True,
                        help="path to output video file")
    # ByteTrack arguments
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value."
                        )
    parser.add_argument("--use_bbox_filters", default=False, action="store_true",
                        help="use ByteTrack bbox size and dimensions ratio filters")
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
    tracker = BYTETracker(args, frame_rate=camera_player.fps)
    frame_id = 0

    video_writer = cv.VideoWriter(video_output_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), camera_player.fps,
                                  (camera_player.width, camera_player.height))
    while True:
        ret, frame = camera_player.get_frame()
        if ret is False:
            break

        outputs, img_info = predictor.inference(frame)
        time_detector = predictor.det_time

        if outputs is not None:
            time_start_tracker = time.time()
            online_targets = tracker.update(outputs,
                                            [img_info['height'], img_info['width']],
                                            predictor.exp.test_size)
            time_tracker = time.time() - time_start_tracker
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if args.use_bbox_filters:
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] < args.min_box_area and vertical:
                        continue
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
            frame_fps = 1. / (time_tracker + time_detector)
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=frame_fps
            )
            if online_im.shape[1] > 1280:
                show_image('frame', online_im, (1280, -1))
            else:
                cv.imshow('frame', online_im)
            video_writer.write(online_im)
        else:
            if img_info['raw_img'].shape[1] > 1280:
                show_image('frame', img_info['raw_img'], (1280, -1))
            else:
                cv.imshow('frame', img_info['raw_img'])
            video_writer.write(img_info['raw_img'])
        cv_key = cv.waitKey(1)
        if cv_key is ord('q'):
            break
        frame_id += 1
    video_writer.release()
    cv.destroyAllWindows()
    print("Finished")
