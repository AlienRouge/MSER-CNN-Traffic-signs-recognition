from Scripts import sign_detection
from Scripts import sign_classification
import cv2

INPUT = r"Input\1.png"
DETECT_MODEL = r"Models\Detection\0.h5"
CLASSIFY_MODEL = r"Models\Classification\1.h5"
IMAGE_SIZE = (96, 96)

image = cv2.imread(INPUT)

regions, images = sign_detection.detect_traffic_signs(image, DETECT_MODEL, IMAGE_SIZE)
sign_classification.classify_areas(image, regions, images, CLASSIFY_MODEL, IMAGE_SIZE)

