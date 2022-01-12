import sys

sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from config import RAFTConfig
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import cv2
import torch
import gc


def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def read_exr(filename, device):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img1, img2, flo):
    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('estimated optical flow')
    ax3.imshow(flo)
    plt.show()


def load_eval_raft():
    config = RAFTConfig(
        dropout=0,
        alternate_corr=False,
        small=False,
        mixed_precision=False
    )

    model = torch.nn.DataParallel(RAFT(config))
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    weights_path = r'D:\Github\RAFT\models\raft-sintel.pth'

    ckpt = torch.load(weights_path, map_location=device)
    print(ckpt)
    model.to(device)
    # model.load_state_dict(ckpt, strict=False)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # ----------------------------
    # SAMPLE FROM DEMO
    n_vis = 3
    #image_files = glob(r'D:\Github\RAFT\demo-frames\*.png')
    # SAMPLE DOC_D
    image_files = glob(r'D:\Github\RAFT\doc_d\*.png')
    image_files = sorted(image_files)

    image_files = sorted(image_files)

    print(f'Found {len(image_files)} images')
    print(sorted(image_files))


    for file1, file2 in tqdm(zip(image_files[:n_vis], image_files[1:1 + n_vis])):
        #image1 = read_exr(file1, device)
        #image2 = read_exr(file2, device)
        image1 = load_image(file1, device)
        image2 = load_image(file2, device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        viz(image1, image2, flow_up)
        gc.collect()

    # ----------------------------
    # SAMPLE VIDEO
    '''
    video_file = r'D:\Github\RAFT\input\how_are_you.mp4'

    cap = cv2.VideoCapture(video_file)

    frames = []
    while True:
        has_frame, image = cap.read()

        if has_frame:
            # image = image[:, :, ::-1]  # convert BGR -> RGB
            frames.append(image)
        else:
            break
    frames = np.stack(frames, axis=0)

    print(f'frame shape: {frames.shape}')

    n_vis = 3

    for i in range(n_vis):
        image1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().to(device)
        image2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().to(device)

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        viz(image1, image2, flow_up)'''


if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_eval_raft()
