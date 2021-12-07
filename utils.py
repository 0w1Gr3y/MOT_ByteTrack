import numpy as np
import cv2 as cv


def resize_image(image, new_size=(-1, 400), interpolation=cv.INTER_LINEAR):
    '''
    :param image: input image
    :param new_size: (width, height) or (-1, h) or (w, -1)
    :param interpolation: opencv cv.INTER_* flag
    :return: resized image
    '''
    image_size = (image.shape[1], image.shape[0])
    if new_size[0] == -1:
        image_size_new = (int(image_size[0] * new_size[1] / image_size[1]), new_size[1])
    elif new_size[1] == -1:
        image_size_new = (new_size[0], int(image_size[1] * new_size[0] / image_size[0]))
    else:
        image_size_new = (new_size[1], new_size[0])
    return cv.resize(image, image_size_new, interpolation=interpolation)  # )


def show_image(image_name, image, new_size=(-1, 400)):
    '''
    :param image_name: image window name
    :param image: input image
    :param new_size: (width, height) or (-1, h) or (w, -1)
    :return:
    '''
    image_show = resize_image(image, new_size)
    cv.imshow(image_name, image_show)
