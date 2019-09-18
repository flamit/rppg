import logging
import pathlib
import cv2
import requests
import matplotlib.pyplot as plt

logger = logging.getLogger('Face Tracking')
MODEL_PATH = pathlib.Path('face/model')
HARR_PATH = MODEL_PATH / 'haarcascade_frontalface_default.xml'
HARR_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
if not HARR_PATH.is_file():
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading harr cascade file')
    r = requests.get(HARR_URL)
    with open(HARR_PATH, 'wb') as f:
        f.write(r.content)


face_cascade = cv2.CascadeClassifier(str(HARR_PATH))


def _get_forehead(x, y, w, h):
    """
    Gets the forehead by looking a the top of the bounds, and in the center.
    """
    w = w / 2
    h = h / 6
    x = x + (w / 2)
    (x, y, w, h) = tuple(map(int, (x, y, w, h)))
    return (list(range(y, h+y)), list(range(x, x+w)))


def get_forehead(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 20))

    return faces
