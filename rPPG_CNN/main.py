import os
import torch
from tqdm import tqdm
from models.sknet.sknet import SKNet
from losses import PearsonLoss
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.rga.resnet import resnet50rga
from models.cbam.resnet_attention import ResidualNet
from models.diaglstm.resnet_diagattn import ResidualNetDiag
from models.attention_aug.resnet_attention_aug import ResidualNetAttn
from models.gsop.gsop import resnet50gsop
from data_reader import FaceFrameReaderTrain, FaceFrameReaderTest

parser = ArgumentParser()
parser.add_argument("--train", default=False, action='store_true', help="Whether training or evaluating")
parser.add_argument("--image_dir", default="images", type=str, help="Directory where images are located")
parser.add_argument("--image_size", default=256, type=int, help="Face image size")
parser.add_argument("--model", default="resnet", type=str, choices=["resnet", "skn", 'attn', 'diag', 'gsop', 'rga'],
                    help="CNN model to use")
parser.add_argument("--T", default=64, type=int, help="Number of frames to stack")
parser.add_argument("--N", default=32, type=int, help="Number of grids to divide the image into")
parser.add_argument("--magnification", default=4, type=int,
                    help="Skin color magnification factor, if 0, no magnification is used.")
parser.add_argument("--batch_size", default=4, type=int, help="Number of inputs in a batch")
parser.add_argument("--n_threads", default=4, type=int, help="Number of workers for data pipeline")
parser.add_argument("--epochs", default=1, type=int, help="Number of complete passes over data to train for")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for the optimizer")
parser.add_argument("--save_dir", default='ckpt', type=str, help="Directory for saving trained models")
parser.add_argument("--save_iter", default=50, type=int, help="Save a model ckpt after these iterations")
parser.add_argument("--ckpt", default='ckpt/checkpoint_0_0.pth', type=str,
                    help="Path to checkpoint to use when testing")


def train(model, args):
    """
    Trains a given model according to specified hyperparameters.
    :param model: The pytorch model to train.
    :param args: The arguments passed into the program from the cli.
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all directories containing face image fames:
    dir_names = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if not x.endswith('.txt')]

    # Initialize the multi-threaded data pipeline
    data_pipeline = FaceFrameReaderTrain(dir_paths=dir_names,
                                         image_size=(args.image_size, args.image_size),
                                         T=args.T,
                                         n=args.N,
                                         magnification=args.magnification)
    data_queue = DataLoader(data_pipeline, shuffle=False, batch_size=args.batch_size, num_workers=args.n_threads)

    # Initialize optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = PearsonLoss(args.T)

    # Send model to training device (gpu or cpu) and set it to train:
    model.to(device)
    model.train()

    # Check if the ckpt save directory exits:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    writer = SummaryWriter()

    # Init training:
    for epoch in range(args.epochs):
        for step, data in enumerate(data_queue):
            opt.zero_grad()
            spatio_tempo, target = data
            logits = model(spatio_tempo.float().to(device))
            loss = loss_function(logits, target.float().to(device))
            if step % args.save_iter == 0:
                print("Epoch: {0}, Step: {1}, Loss: {2}".format(epoch, step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), epoch * len(data_queue) + step)
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, "checkpoint_{0}_{1}.pth".format(epoch, step)))
            loss.backward()
            opt.step()


def predict(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_pipeline = FaceFrameReaderTest(args.image_dir, (args.image_size, args.image_size), args.T, args.N,
                                        args.magnification)
    data_queue = DataLoader(data_pipeline, shuffle=False, batch_size=1, num_workers=1)

    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for data in tqdm(data_queue):
            preds = model(data.float().to(device)).view(-1)
            predictions.extend(preds.cpu().numpy())

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    with open('outputs/prediction.txt', 'w') as file_handler:
        for item in predictions:
            file_handler.write("{}\n".format(item))

    print("Done! Prediction trace output written in outputs/prediction.txt")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model == "resnet":
        model = ResidualNet("ImageNet", 50, args.T, 'CBAM')
    elif args.model == "skn":
        model = SKNet(args.T, [3, 4, 6, 3])
    elif args.model == "attn":
        size = (args.image_size // args.N) ** 2
        model = ResidualNetAttn(args.T, size)
    elif args.model == "diag":
        model = ResidualNetDiag("ImageNet", 50, args.T, "BAM")
    elif args.model == "gsop":
        attn_pos = [['0'] * 2 + ['1'], ['0'] * 3 + ['1'], ['0'] * 22 + ['1'], ['0'] * 3]  # Mode 2
        model = resnet50gsop(False, attn_pos, 128, GSoP_mode=2, num_classes=args.T)
    elif args.model == "rga":
        h = (args.image_size // args.N) ** 2
        w = args.T
        model = resnet50rga(num_classes=args.T, dims=(h, w))
    else:
        raise ValueError("Model name provided is invalid")
    if args.train:
        train(model, args)
    else:
        predict(model, args)
