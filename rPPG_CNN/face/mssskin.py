import os
import cv2
from face.face_tracking import get_forehead
from tqdm import trange


def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_and_write_face(video_path, write_dir, T=100):
    cap = cv2.VideoCapture(video_path)
    if T == 0:
        T = number_of_frames(video_path)
    rot = 90
    ret, frame = cap.read()
    for tries in range(3):
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        dst = cv2.warpAffine(frame, M, (cols, rows))
        dims = get_forehead(dst)
        if len(dims) == 0:
            print("Face not detected, rotating and trying again...")
            rot += 90
        else:
            print("Video rotation found to be {} degrees".format(rot))
            break
    if rot == 360:
        print("No face detected in the video")
        raise

    for i in trange(T):
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        dst = cv2.warpAffine(frame, M, (cols, rows))
        dims = get_forehead(dst)
        if len(dims) > 0:
            x, y, w, h = dims[0]
            forehead_img = dst[y:y+h, x:x+h]

            cv2.imwrite(os.path.join(write_dir, '{0}.png'.format(i)), forehead_img)
        ret, frame = cap.read()

