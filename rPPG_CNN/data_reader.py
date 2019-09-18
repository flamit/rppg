from torch.utils.data import Dataset
from skimage.measure import block_reduce
import numpy as np
import pandas as pd
import csv
import cv2
import os


class FaceFrameReaderTrain(Dataset):
    """Face frame reader for training the rPPG CNN"""

    def __init__(self, dir_paths, image_size, T=100, n=16, magnification=0):
        """
        Initializes the data reader for training.
        :param dir_paths: Path to directory containing directories of images of each video file.
        :param image_size: Tuple specifying the height and width of the face images to use for training.
        :param T: Integer specifying the number of frames to use for training.
        :param n: Integer specifying the block size to use for averaging the image.
        """
        self.dir_paths = dir_paths
        self.image_names = [sorted([x for x in os.listdir(y)], key=lambda x: int(os.path.splitext(x)[0])) for y in
                            dir_paths]
        self.image_size = image_size
        self.T = T
        self.max_idx = [len(x) - T for x in dir_paths]
        self.n = n
        self.count = 0
        self.magnification = magnification
        for image_names in self.image_names:
            self.count += len(image_names)

    def read_gt_file(self, idx):
        """
        Reads a ground truth file in the list of paths at location idx.
        :param idx: The location in the list of gt file paths to read
        :return: An array of BVP trace values read from the gt file.
        """
        gt_file = self.dir_paths[idx] + '.txt'
        with open(gt_file, 'r') as input_file:
            for row_idx, row in enumerate(csv.reader(input_file, delimiter=',')):
                if len(row) > 0:
                    if "B: BVP" in str(row):
                        header_row = row_idx

        data = pd.read_csv(gt_file, skiprows=header_row + 2, header=None, delimiter=',')
        data = np.asarray(data[1].values.data)
        return data

    def __len__(self):
        """Returns an estimate of the length of the dataloader queue,
        Not really important since our training selection is random and
        train time can be increased by increasing epochs."""
        return self.count

    def __getitem__(self, idx):
        """
        Reads T consequetive face frames, and converts them into the required rPPG input.
        image -> YUV color space -> resize -> average nxn blocks in the image -> flatten into a row
        -> Stack T frames -> convert to channels first for pytorch.
        Loads the gt file and takes an average of 8 values to compress the 256 bit rate into roughly the same
        30fps rate for the video.
        :param idx: The video file images to read.
        :return: A batch of rPPG input signals along with gt.
        """
        frames = []
        idx = np.random.randint(0, len(self.dir_paths))
        gt = self.read_gt_file(idx)
        start_frame_idx = np.random.randint(0, len(self.image_names[idx]) - self.T)
        for i in range(start_frame_idx, start_frame_idx + self.T):
            path = os.path.join(self.dir_paths[idx], self.image_names[idx][i])
            image = cv2.imread(path, 1)
            if self.magnification:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                cr_mean = np.mean(image[..., 1])
                cb_mean = np.mean(image[..., 2])
                image[..., 1] = cr_mean + (image[..., 1] - cr_mean) * self.magnification
                image[..., 2] = cb_mean + (image[..., 2] - cb_mean) * self.magnification
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, self.image_size)
            image = block_reduce(image, (self.n, self.n, 1), np.mean)
            image = np.reshape(image, [1, -1, 3])
            frames.append(image)
        images_stacked = np.concatenate(frames)
        images_stacked = np.transpose(images_stacked, [2, 1, 0])
        gt = block_reduce(gt, (8,), np.mean)[start_frame_idx: start_frame_idx + self.T]
        return images_stacked, gt


class FaceFrameReaderTest(Dataset):
    """Face frame reader for Testing/Predictions only."""

    def __init__(self, image_dir, image_size, T=100, n=16, magnification=0):
        """
        Initializes the data reader for training.
        :param image_dir: Path to directory containing images to predict on.
        :param image_size: Tuple specifying the height and width of the face images to use for training.
        :param T: Integer specifying the number of frames to use for training.
        :param n: Integer specifying the block size to use for averaging the image.
        """
        self.image_dir = image_dir
        self.image_names = sorted([x for x in os.listdir(self.image_dir) if x.endswith('.png')],
                                  key=lambda x: int(os.path.splitext(x)[0]))
        if len(self.image_names) == 0:
            raise Exception("No images files found in the specified directory")
        self.image_size = image_size
        self.T = T
        self.n = n
        self.count = 0
        self.magnification = magnification

    def __len__(self):
        return int(len(self.image_names) / self.T)

    def __getitem__(self, idx):
        frames = []
        idx = idx * self.T
        for i in range(idx, idx + self.T):
            path = os.path.join(self.image_dir, self.image_names[i])
            image = cv2.imread(path, 1)
            if self.magnification:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                cr_mean = np.mean(image[..., 1])
                cb_mean = np.mean(image[..., 2])
                image[..., 1] = cr_mean + (image[..., 1] - cr_mean) * self.magnification
                image[..., 2] = cb_mean + (image[..., 2] - cb_mean) * self.magnification
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, self.image_size)
            image = block_reduce(image, (self.n, self.n, 1), np.mean)
            image = np.reshape(image, [1, -1, 3])
            frames.append(image)
        images_stacked = np.concatenate(frames)
        images_stacked = np.transpose(images_stacked, [2, 1, 0])

        return images_stacked
