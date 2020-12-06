from imutils.video import filevideostream
from mtcnn.mtcnn import MTCNN
import cv2

img = cv2.imread('faces/' + "1.jpg")
detector = MTCNN()


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = filevideostream.FileVideoStream('Montreal.m4v').start()
face = 1
while True:
    # Capture frame-by-frame
    frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        thumb = frame[y:y + h, x:x + w]
        d = detector.detect_faces(thumb)
        if d == []:
            cv2.imwrite('faces/' + str(face) + '.jpg', thumb)
        else:
            cv2.imwrite('non-faces/' + str(face) + '.jpg', thumb)
        face += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    cv2.waitKey(1)

