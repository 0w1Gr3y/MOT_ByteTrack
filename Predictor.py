import time
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.data.data_augment import ValTransform
from loguru import logger
import torch
from YOLOX.yolox.utils import fuse_model, postprocess, vis


class Predictor(object):
    def __init__(
            self,
            ckpt,
            name,
            fuse=False,
            device="gpu",
            fp16=False,
            legacy=False,
    ):
        self.exp = get_exp(None, name)
        self.exp.test_conf = 0.1
        self.exp.nmsthre = 0.4
        self.exp.test_size = (640, 640)
        model = self.exp.get_model()
        model.cuda()
        model.eval()

        ckpt_file = ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        if fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        self.model = model
        self.cls_names = COCO_CLASSES

        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            self.det_time = time.time() - t0
            outputs = self.filter_classes(outputs[0])
        return outputs, img_info

    def filter_classes(self, outputs):
        if outputs is None:
            return None
        output_mask = []
        for output in outputs:
            cls = output[6]
            cls = int(cls)
            if self.cls_names[cls] in ['car', 'bus', 'truck', 'person', 'bicycle', 'motorcycle']:
                output_mask.append(True)
            else:
                output_mask.append(False)
                continue
        if output_mask.count(False) == 0:
            return None
        else:
            return outputs[output_mask]

    def visual(self, output, img_info):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, self.confthre, self.cls_names)
        return vis_res