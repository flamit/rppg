from face.mssskin import extract_and_write_face
from argparse import ArgumentParser
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument("--video_dir", type=str, default='videos', help="Directory video files are located")
parser.add_argument("--image_dir", type=str, default='images',
                    help="Directory where face frames extracted from videos are located")
parser.add_argument("--max_frames", type=int, default=0,
                    help="Max number of frames to extract from a video, set to 0 to extract all frames")


def get_video_filenames(video_dir):
    video_files = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
    return video_files


if __name__ == '__main__':
    args = parser.parse_args()
    video_filenames = get_video_filenames(args.video_dir)
    pbar = tqdm(video_filenames)
    for video_filename in pbar:
        pbar.set_description("Processing video file: {}".format(video_filename))
        fn = os.path.basename(video_filename)
        idx = fn.index('.')
        fn = fn[:idx]
        work_dir = os.path.join(args.image_dir, fn)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        extract_and_write_face(video_filename, work_dir, args.max_frames)