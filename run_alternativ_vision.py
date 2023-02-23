import argparse
import cv2
import requests

from utils.audio.synthesize_sound import SoudSynthesizer
from utils.camera_calibration.camera_calibration_utils import im_point_to_angles_inter
from utils.mot.tracker_yolov8 import Tracker


class AlternativeVision:
    """
    A class that defines an engine for binaural sound generation in real time.
    It contains a multi-object detection and tracking model based on YOLOv8 and StrongSort
    used to track the position of objects in the environment (for now only ('cup', 'keyboard, 'mouse')).
    It also contains a sound synthesizer with creates a 3D sound for each object based on its
    position determined by the tracking model.
    """
    def __init__(self, video_source=0,  yolo_im_sz=(640, 640),
                 save_video=False, video_out_path="output.avi", cam_fov=(160, 160),
                 visualize_detections=True, **kwargs) -> None:

        self.save_video = save_video
        self.cam_fov = cam_fov
        self.visualize_detections = visualize_detections

        self.cap = cv2.VideoCapture(video_source)
        ret, frame = self.cap.read()

        self.src_height, self.src_width, _ = frame.shape
        # define tracker
        self.detection_im_sz = yolo_im_sz
        self.object_tracker = Tracker(imgsz=yolo_im_sz)

        # define a binaural sound synthesizer
        self.sound_synthesizer = SoudSynthesizer(img_size=[self.src_height, self.src_width])
        # run synthesizer in a separate thread
        self.sound_synthesizer.run()
        
        if save_video:

            # video writer
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.vid_out = cv2.VideoWriter(video_out_path, self.fourcc, 20.0, (800, 800))

    def run(self):

        ret = True
        while ret:
            ret, frame = self.cap.read()
            
            if not ret:
                exit() 

            # get image size (for the case it changes in runtime)
            self.src_height, self.src_width, _ = frame.shape

            # detect objects
            detected_objects = self.object_tracker.track_objects(frame)

            # compute the directions (elevation and azimuth) of the detected objects
            for id, obj_dict in detected_objects.items():
                obj_dict['u_left'] = obj_dict['u_left'] * (self.src_width / self.detection_im_sz[1]) 
                obj_dict['v_top'] = obj_dict['v_top'] * (self.src_height / self.detection_im_sz[0]) 
                obj_dict['u_right'] = obj_dict['u_right'] * (self.src_width / self.detection_im_sz[1]) 
                obj_dict['v_bott'] = obj_dict['v_bott'] * (self.src_height / self.detection_im_sz[0]) 
                obj_dict['class_id'] = obj_dict['class_id']
                angle_u, angle_v = im_point_to_angles_inter([(obj_dict['u_left']+obj_dict['u_right'])/2,
                                                             (obj_dict['v_top']+obj_dict['v_bott'])/2],
                                                            (self.src_width, self.src_height), self.cam_fov)

                obj_dict['elevation'] = angle_v
                obj_dict['azimuth'] = angle_u
                        
                detected_objects[id] = obj_dict
            # update sound synthesizer
            self.sound_synthesizer.update_synthesizer(detected_objects)

            # visualize detections
            if self.visualize_detections:
                for id, obj_dict in detected_objects.items():
                    
                    if obj_dict['class_id'] not in ['cup', 'keyboard']:
                        continue 
                    # self.sound_synthesizer.play_sound(angle_v, angle_u)
                    cv2.rectangle(frame, (round(obj_dict['u_left']), round(obj_dict['v_top'])), 
                                        (round(obj_dict['u_right']), round(obj_dict['v_bott'])), 
                                        (255,0,0), 1)
                    cv2.putText(frame, f"{obj_dict['class_id']}: {int(id)}", 
                                (round(obj_dict['u_left']), round(obj_dict['v_top'])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("detections", cv2.resize(frame, (800, 800)))
            # self.vid_out.write(cv2.resize(frame, (800, 800)))
            cv2.waitKey(10)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-source', type=str, default=0)
    parser.add_argument('--yolo-im-sz', type=int, default=(640, 640))
    parser.add_argument('--save-video', type=bool, default=False)
    parser.add_argument('--visualize-detections', type=bool, default=True)
    parser.add_argument('--video-out-path', type=str, default='vid_out.avi')
    parser.add_argument('--use-cam-webserver', type=bool, default=False)
    parser.add_argument('--cam-webserver-url', type=str, default='http://192.168.0.171')
    parser.add_argument('--cam-webserver-port', type=str, default='81')
    args = parser.parse_args()

    # set up ESP32-CAM cam webserver url
    if args.use_cam_webserver:
        requests.get(f"{args.cam_webserver_url}/control?var=framesize&val=10?",
                     timeout=5)  # set resolution of ESP32-CAM to 800x600
        video_stream = args.cam_webserver_url + f":{args.cam_webserver_port}/stream"
        args.video_source = video_stream

    return args


if __name__ == "__main__":
    # parse arguments
    opts = parse_args()
    # initialize alternative vision engine
    alternative_vision = AlternativeVision(**vars(opts))
    # run alternative vision engine
    alternative_vision.run()
