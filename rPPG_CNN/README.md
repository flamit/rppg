# rPPG_CNN

This repository hosts a CNN model for remote heart rate sensing using frames from a user's face as an input.

# Pre-requisites
To install all the required python packages, use the requirements.txt file. It is assumed you have Python (preferably Anaconda) already installed:
```bash
pip install -r requirements.txt
```
# Supported models:
Currently, the following models can be trained in the rPPG framework:
1. "resnet" for a resnet of depth 50.
2. "skn" for selective kernel networks of depth 50.
3. "attn" for a resnet50 with each 3x3 convolution replaced by an attention augmented convolution block.
4. "diag" for a resnet50 with channel attention + attentionRNN.
5. "gsop" for a resnet50 with Global Second Order Pooling (mode 2).
6. "rga" for a resnet50 with Relation aware global pooling. This mode is currently untested, USE WITH CAUTION. To use this model, please ensure that the "image_size", "T" and "N" parameters are set in such a way that the input shape is at least 128x128. Anything smaller cannot be processed by the RGA module. The two spatial dimensions can also be unequal, this is allowed, as long as they are greater than 128.

# Training
To train the CNN, there are 3 steps:
1. Pre-processing the video to convert it into an image dataset.
2. Placing ground truth data.
3. Train the CNN on the image dataset obtaines from step 2.

# Pre-process the video files:
Your video files should all be inside one folder and there should be nothing else in that folder. This step only needs to be done once. It is time consuming, but makes training, prototyping and testing the model afterwords fast. So once done, this step should not be repeated again, unless there are newer video files to convert.

Run the following command in your terminal: 
```bash
python data_preprocess.py --video_dir=<path to video folder> --images_dir=<path where to save face images> --max_frames=0
```
If the video is rotated, the script automatically checks all rotations that are multiples of 90 for a face, whichever is found to contain a face, is used as the rotation fix for all frames.

The ```max_frames=0``` command can be used to limit the number of frames to extract from the video to cut running time on the script (if the video is too long). Setting it to 0 means all frames are extracted from the video automatically.

# Place the ground truth data files:
Once the video processing is done, make sure that the ground truth files for each of the video have the SAME NAME as the video. Put all these files in the folder you specified as ```---images_dir```.

# Training:
Now training can be run. To start training the CNN, simply do:
```bash
python main.py --train --model=<model name> --image_dir=<path to images dir> --image_size=256 --T=64 --N=32 --batch_size=32 --n_threads=4 --epochs=5 --lr=1e-3 --save_iter=200 --magnification=0
```

If you set "--magnification" to any value greater than 0, skin color magnification will be used as the magnification factor. Setting it to 0 turns skin color magnification off.

If you followed all the previous steps correctly, training should now run without problems. To read description of what each of the command line options do, read the "help" parameter in the main.py lines 11-21.

# Prediction/Testing:
To run a trained model on a set of images:
```bash
python main.py --model=<model name> --image_dir=<path to directory containing images> --image_size=256 --T=64 --N=32 --ckpt=<path to saved checkpoint>
```
It is critical the image_size, T and N parameters are the same as when they were during training, since they define the CNN input and output shapes, changes in these parameters between train and test will cause the checkpoint weights to not load correctly. The output is saved in a text file in the folder "outputs" created in the project directory.

# Note:
The extracted face frames from the video are named using numbers because they can be sorted and mapped to the ground truth values using those numbers. Changing the naming schema of the image files will cause this mapping to break and should be kept in mind. Also, it is assumed that all frames in the video contain a face which is extracted, long sequences of missing faces will also break this mapping. It is also assumed that the videos are 30fps and ground truth is sampled at 256 samples a second (change in these will require a modification in the algorithmic logic).

# Tensorboard:
Tensorboardx allows us to save training run data in PyTorch. To monitor the loss function as training progresses, please start tensorboard by running the following script in the terminal:
```bash
tensorboard --logdir=<path to tensorboardx log dir>
```
Then open up your browser and navigate to "localhost:6006" to display the visualizations.

# TODOs:
1. Bug fixes.
2. ~~Make data pipeline better.~~
3. ~~Make face detector robust to rotation.~~
4. ~~Make the predict function run correctly.~~
5. Maybe changes in the algorithmic logic.
6. ~~Verify the rPPG input signal as being correct.~~
