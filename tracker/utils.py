import cv2 as cv
import numpy as np


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv.putText(im, id_text, (intbox[0], intbox[1]), cv.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
                   thickness=text_thickness*3, lineType=cv.LINE_AA)
        cv.putText(im, id_text, (intbox[0], intbox[1]), cv.FONT_HERSHEY_PLAIN, text_scale, color,
                   thickness=text_thickness, lineType=cv.LINE_AA)

    cv.putText(im, 'Frame: %d FPS: %.2f Num: %d' % (frame_id, fps, len(tlwhs)),
               (0, int(15 * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=6, lineType=cv.LINE_AA)
    cv.putText(im, 'Frame: %d FPS: %.2f Num: %d' % (frame_id, fps, len(tlwhs)),
               (0, int(15 * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2, lineType=cv.LINE_AA)
    '''
    # congestion warning
    your_threshold = 25
    if len(tlwhs) >= your_threshold:
    	cv.putText(im, 'congestion warning!', (0, int(30 * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=6, lineType=cv.LINE_AA)
	cv.putText(im, 'congestion warning!', (0, int(30 * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
    '''
    return im
