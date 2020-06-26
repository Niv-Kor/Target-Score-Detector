from VideoAnalyzer import VideoAnalyzer
from Sketcher import Sketcher
import cv2

# input
model = cv2.imread('res/input/target.jpg')
video_name = 'res/input/video.mp4'
bullseye_point = (325,309)
inner_diameter_px = 50
inner_diameter_inch = 1.5
rings_amount = 6
display_in_cm = False

# get a sample frame from the video
cap = cv2.VideoCapture(video_name)
_, test_sample = cap.read()

# calculate the sizes of the frame and the input
model_h, model_w, _ = model.shape
frame_h, frame_w, _ = test_sample.shape
pixel_to_inch = inner_diameter_inch / inner_diameter_px
pixel_to_cm = pixel_to_inch * 2.54
measure_unit = pixel_to_cm if display_in_cm else pixel_to_inch
measure_unit_name = 'cm' if display_in_cm else '"'

# analyze
sketcher = Sketcher(measure_unit, measure_unit_name)
video_analyzer = VideoAnalyzer(video_name, model, bullseye_point, rings_amount, inner_diameter_px)
video_analyzer.analyze('res/output/output.mp4', sketcher)