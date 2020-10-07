import cv2
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import configparser




def get_gray_img(frame, data, is_square, img_width, img_height):
    x, y, w, h = data
    img = Image.fromarray(frame[y:y+h, x:x+w])
    if is_square:
        return resizeimage.resize_cover(img, [img_width, img_height])
    else:
        width_scaled_img = resizeimage.resize_width(img, img_width)
        return resizeimage.resize_crop(width_scaled_img, [img_width, img_height])



def resize_face_rectangle(face_data, img_width, img_height):
    scale_factor = img_height/img_width if img_height > img_width else img_width/img_height
    x, y, w, h = face_data
    # new height should be distributed equally between starting and ending point
    if img_height > img_width:
        scaled_height = h*scale_factor
        height_diff = int((scaled_height-h)/2) + 1
        y -= height_diff
        h += height_diff
    else:
        scaled_width = w*scale_factor
        width_diff = int((scaled_width-w)/2) + 1
        x -= width_diff
        w += width_diff
    return x, y, w, h


def face_recognition(name='default', path='./', pictures=1):

    config = configparser.ConfigParser()
    config.read('configuration.ini')
    img_height = config.getint('IMAGES_DATA', 'HEIGHT')
    img_width = config.getint('IMAGES_DATA', 'WIDTH')
    color = (0, 255, 0)  # BGR => green
    thickness = 2

    is_square_ratio = img_height == img_width

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    if video_capture is None or not video_capture.isOpened():
        raise Exception('Missing webcam')
    if pictures < 1:
        raise Exception('needs at least 1 picture')
    amount_of_pictures = 0
    face_data = None
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
            # resize for larger desired ratio
            face_data = [x, y, w, h]
            if not is_square_ratio:
                x, y, w, h = resize_face_rectangle(face_data, img_width, img_height)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            face_data = [x, y, w, h]

        if cv2.waitKey(1) & 0xFF == ord('s') and face_data is not None:
            face_gray = get_gray_img(frame_gray, face_data, is_square_ratio, img_width, img_height)
            face_gray.save(
                f'{path}/{name}_{amount_of_pictures+1}.pgm', face_gray.format)
            amount_of_pictures += 1
            if amount_of_pictures == pictures:
                break

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return np.array(face_gray)
