# The code in the script is mostly copied from the tracker defined 'yolov8_tracking'
# The code was modified to turn the function 'run' in 'yolov8_tracking/track.py' into a class
# and strip down to the stuff needed for this project

import os
import cv2

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov8_tracking' # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker


class Tracker:
    def __init__(
            self,
            source='0',
            yolo_weights=WEIGHTS / 'yolov8n.pt',  # model.pt path(s),
            reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
            tracking_method='strongsort',
            imgsz=(480, 640),  # inference size (height, width)
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            save_txt=False,  # save results to *.txt
            project=ROOT / 'runs/track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            eval=False,  # run multi-gpu eval
    ):

        source = str(source)
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + reid_weights.stem
        self.save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'tracks' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if eval:
            device = torch.device(int(device))
        else:
            device = select_device(device)
        self.model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_imgsz(imgsz, stride=stride)  # check image size

        # only one source for now
        nr_sources = 1

        # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
        tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')

        # Create as many strong sort instances as there are video sources
        self.tracker_list = []
        for i in range(nr_sources):
            tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
            self.tracker_list.append(tracker, )
            if hasattr(self.tracker_list[i], 'model'):
                if hasattr(self.tracker_list[i].model, 'warmup'):
                    self.tracker_list[i].model.warmup()
        self.outputs = [None] * nr_sources

        # Run tracking
        self.model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources

        # detected objects
        self.detected_objects = {}

    @torch.no_grad()
    def track_objects(self,
                      im,
                      conf_thres=0.25,  # confidence threshold
                      iou_thres=0.45,  # NMS IOU threshold
                      max_det=1000,  # maximum detections per image
                      device='cuda',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                      show_vid=False,  # show results
                      save_txt=False,  # save results to *.txt
                      save_crop=False,  # save cropped prediction boxes
                      save_vid=False,  # save confidences in --save-txt labels
                      nosave=False,  # do not save images/videos
                      classes=None,  # filter by class: --class 0, or --class 0 2 3
                      agnostic_nms=False,  # class-agnostic NMS
                      augment=False,  # augmented inference
                      visualize=False,  # visualize features
                      line_thickness=2,  # bounding box thickness (pixels)
                      hide_labels=False,  # hide labels
                      hide_conf=False,  # hide confidences
                      hide_class=False,  # hide IDs
                      half=False,  # use FP16 half-precision inference
                      ):
        im = cv2.resize(im, self.imgsz)
        im0s = [np.copy(im)]
        im = np.transpose(im, (2, 0, 1))
        im = np.expand_dims(im, axis=0)
        # im0 = im0s

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        # Forget old detections
        self.detected_objects = {}
        for i, det in enumerate(pred):  # detections per image
            self.seen += 1
            im0 = im0s[i].copy()
            self.curr_frames[i] = im0

            # txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
            s = '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(np.ascontiguousarray(im0), line_width=line_thickness, pil=not ascii)

            if self.prev_frames[i] is not None and self.curr_frames[i] is not None:  # camera motion compensation
                if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
                    self.tracker_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization
                if len(self.outputs[i]) > 0:
                    for j, (output) in enumerate(self.outputs[i]):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]

                        self.detected_objects[int(id)] = {
                            'u_left': round(bbox_left),
                            'u_right': round(bbox_left + bbox_w),
                            'v_top': round(bbox_top),
                            'v_bott': round(bbox_top + bbox_h),
                            'class_id': self.names[int(cls)]
                        }

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {self.names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)

            self.prev_frames[i] = self.curr_frames[i]
            return self.detected_objects


