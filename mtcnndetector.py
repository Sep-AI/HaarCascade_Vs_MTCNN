from mtcnn.mtcnn import MTCNN
import cv2
img = cv2.imread('faces/' + "1.jpg")
detector = MTCNN()
print(detector.detect_faces(img))
