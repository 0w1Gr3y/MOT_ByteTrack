from VideoPlayer import VideoPlayer
import cv2 as cv
from Predictor import Predictor
from tracker.byte_tracker import BYTETracker
from tracker.utils import plot_tracking
import argparse
import time

# python .\main.py --video_input assets/new/KarolMajek720.avi --video_output output_yolox.avi
if __name__ == '__main__':
    parser = argparse.ArgumentParser("ByteTrack Demo!")
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

    args = parser.parse_args()

    predictor = Predictor(ckpt='weights/yolox_m.pth', name='yolox-m')
    video_input_path = args.video_input
    video_output_path = args.video_output
    # video_name = 'new\karol4k.mp4'
    # video_name = 'new\KarolMajek720.avi'
    # --video_input assets/new/KarolMajek720.avi --video_output results/output_yolox.avi
    # --video_input MNn9qKG2UFI.webm --video_output results/output_yolox.avi

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
            cv.imshow('frame', online_im)
            video_writer.write(online_im)
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
