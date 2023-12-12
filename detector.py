from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2

class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # load model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def open_video(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened() == False): 
            print("Error opening video")
            return
        else:
            (sucess, img) = cap.read()
            return sucess, img, cap
        
    def print_objects(self, img):
        outputs = self.predictor(img)
        print("Objects:")
        
        classes = outputs["instances"].pred_classes
        print(classes)

        boxes = outputs["instances"].pred_boxes
        print(boxes)

        return outputs
    
    def is_object_moved(self, distance, obj_1, obj_2):
        first = obj_1.get_centers()[0]
        second = obj_2.get_centers()[0]
        second_min = [value-distance for value in second]
        second_max = [value+distance for value in second]

        for i, value in enumerate(first):
            if second_min[i] < first[i] and first[i] < second_max[i]:
                return True
            return False
        
    def compare_objects(self, outputs_1, outputs_2):
        # list of moved objects
        outputs_list = [] 

        classes_1 = outputs_1["instances"].pred_classes
        boxes_1 = outputs_1["instances"].pred_boxes
        classes_2 = outputs_2["instances"].pred_classes
        boxes_2 = outputs_2["instances"].pred_boxes

        for i1, item1 in enumerate(classes_1):
            print("--- Comparing ---")
            for i2, item2 in enumerate(classes_2):
                if item2 == item1:
                    obj_1 = boxes_1[i1]
                    obj_2 = boxes_2[i2]
                    print(f"{item1} -> {obj_1} - {obj_1.get_centers()}")
                    print(f"{item2} -> {obj_2} - {obj_2.get_centers()}")

                    # check if the object didn't move (distance less than 3)
                    condition = self.is_object_moved(3, obj_1, obj_2)
                    if condition:
                        print("Object isn't moved.")

                    else: # check if the object moved (distance between 3 and 20)
                        condition = self.is_object_moved(20, obj_1, obj_2)
                        if condition:
                            print("Object is moved.")
                            outputs_list.append(outputs_2["instances"][i2])
        return outputs_list

    def prediction(self, sucess, img, cap, outputs_1):
        while sucess:
            # print recognised objects
            outputs_2 = self.print_objects(img)

            # compare objects from the previous image with objects from the present image
            # list with objects which were moved
            outputs_list = self.compare_objects(outputs_1, outputs_2)

            # if any object moved -> show the result
            if len(outputs_list) > 0:
                h, w, c = img.shape
                new_out = Instances(image_size=[h,w])
                new_instances = new_out.cat(outputs_list)

                viz = Visualizer(img[:,:,::-1],
                                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                out = viz.draw_instance_predictions(new_instances.to("cpu"))
                cv2.imshow("Result", out.get_image()[:,:,::-1])

            else:
                cv2.imshow("Result", img)

            # the present image becomes the previous image
            outputs_1 = outputs_2

            # read the next image
            (sucess, img) = cap.read() 

            # press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break