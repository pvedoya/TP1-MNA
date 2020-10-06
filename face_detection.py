import cv2
from PIL import Image
from resizeimage import resizeimage
import numpy as np
img_height = 112
img_width = 92
color = (0, 255, 0)  # BGR => green
thickness = 2


def get_gray_img(frame, data):
    x, y, w, h = data
    img = Image.fromarray(frame[y:y+h, x:x+w])
    width_scaled_img = resizeimage.resize_width(img, img_width)
    return resizeimage.resize_crop(width_scaled_img, [img_width, img_height])


def face_recognition(name='default', path='./', pictures=1):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    if video_capture is None or not video_capture.isOpened():
        raise Exception('Missing webcam')
    if pictures < 1:
        raise Exception('needs at least 1 picture')
    amount_of_pictures = 0
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            # higher number => less sensitive. Big number good for only one face.
            minSize=(int(img_width*2), int(img_height*2))
        )

        # Draw a rectangle around the faces
        # x and y are the position of the top left point of the rectangle around the face
        # with 0,0 being the top left point of the camera view
        # w and h are the width and height of the rectangle around the face
        for (x, y, w, h) in faces:
            # resize height to get a 92:112 aspect ratio
            scale_factor = img_height/img_width
            scaled_height = h*scale_factor
            # new height should be distributed equally between starting and ending point
            height_diff = int((scaled_height-h)/2) + 1
            y -= height_diff
            h += height_diff
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            face_data = [x, y, w, h]


        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            face_gray = get_gray_img(frame_gray, face_data)
            face_gray = get_gray_img(frame_gray, face_data)
            face_gray.save(
                f'{path}/{name}_{amount_of_pictures+1}.pgm', face_gray.format)
            amount_of_pictures += 1
            if amount_of_pictures == pictures:
                break
        # Display the resulting frame
        cv2.imshow('Video', frame)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return np.array(face_gray)

# face_recognition(name = 'santi', path='./att_faces/santi', pictures=10) # use this to take the training pictures
# use this to take the to_match picture
# face_recognition(name='santi', pictures=1)
