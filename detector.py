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
        if (not cap.isOpened()): 
            print("Error opening video")
            return
        else:
            sucess, img = cap.read()
            return sucess, img, cap
        
    def print_objects(self, img):
        objects = self.predictor(img)
        print("Objects:")
        
        classes = objects["instances"].pred_classes
        print(classes)

        boxes = objects["instances"].pred_boxes
        print(boxes)

        return objects
    
    def is_object_moved(self, distance, previous_obj_box, present_obj_box):
        previous_obj_center = previous_obj_box.get_centers()[0]
        present_obj_center = present_obj_box.get_centers()[0]
        obj_center_min = [value-distance for value in present_obj_center]
        obj_center_max = [value+distance for value in present_obj_center]

        for i, value in enumerate(previous_obj_center):
            if obj_center_min[i] < previous_obj_center[i] < obj_center_max[i]:
                return True
            return False
        
    def compare_objects(self, previous_objects, present_objects):
        list_of_moved_objects = [] 

        previous_classes = previous_objects["instances"].pred_classes
        previous_boxes = previous_objects["instances"].pred_boxes
        present_classes = present_objects["instances"].pred_classes
        present_boxes = present_objects["instances"].pred_boxes

        for i1, item1 in enumerate(previous_classes):
            print("--- Comparing ---")
            for i2, item2 in enumerate(present_classes):
                if item2 == item1:
                    previous_obj_box = previous_boxes[i1]
                    present_obj_box = present_boxes[i2]
                    print(f"{item1} -> {previous_obj_box} - {previous_obj_box.get_centers()}")
                    print(f"{item2} -> {present_obj_box} - {present_obj_box.get_centers()}")

                    # check if the object didn't move (distance less than 3)
                    if self.is_object_moved(3, previous_obj_box, present_obj_box):
                        print("Object isn't moved.")

                    # check if the object moved (distance between 3 and 20)
                    else:
                        if self.is_object_moved(20, previous_obj_box, present_obj_box):
                            print("Object is moved.")
                            list_of_moved_objects.append(present_objects["instances"][i2])

                        else: print("Different objects")

        return list_of_moved_objects

    def prediction(self, sucess, img, cap, previous_objects):
        while sucess:
            present_objects = self.print_objects(img)
            list_of_moved_objects = self.compare_objects(previous_objects, present_objects)

            # if any object moved -> show the result
            if len(list_of_moved_objects) > 0:
                h, w, c = img.shape
                new_out = Instances(image_size=[h,w])
                moved_objects = new_out.cat(list_of_moved_objects)

                viz = Visualizer(img[:,:,::-1],
                                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_instance_predictions(moved_objects.to("cpu"))
                cv2.imshow("Result", output.get_image()[:,:,::-1])

            else:
                cv2.imshow("Result", img)

            # the present image becomes the previous image
            previous_objects = present_objects

            # read the next image
            sucess, img = cap.read() 

            # press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break