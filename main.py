from detector import Detector

detector = Detector()
sucess, img, cap = detector.open_video("video.mp4")
first_objects = detector.print_objects(img)
detector.prediction(sucess, img, cap, first_objects)
