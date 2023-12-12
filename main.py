from detector import Detector

detector = Detector()
sucess, img, cap = detector.open_video("szosty.mp4")
outputs = detector.print_objects(img)
detector.prediction(sucess, img, cap, outputs)
