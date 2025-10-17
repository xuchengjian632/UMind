import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import argparse
from PIL import Image
import numpy as np
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
parser.add_argument('--result_path', default='./results/generation/' , type=str)
parser.add_argument('--test_image_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images' , type=str)
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--alpha', default=0.5, type=float, help='parameters for balancing EEG_image and EEG_text')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--model_type', default='ViT-H-14', type=str)
parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)
args = parser.parse_args()

sub = 'sub-08'
import utils

seed_n = args.seed
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

# Define the source and target directories
source_dir = "/media/siat/disk1/code/EEG-to-image/results/generated_imgs/sub-08/only_image_pred_mindeye_1.0"
# source_dir = '/media/siat/disk1/code/EEG_Image_decode-main/Generation/generated_imgs/sub-08/clip_embeds_and_low_level_strength0.5_steps10'
target_dir = f"./results/models/{args.encoder_type}/{args.val}/{sub}"

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize a list to hold all the image tensors
tensor_list = []

# Iterate over the folders in the source directory
for folder_name in sorted(os.listdir(source_dir)):
    folder_path = os.path.join(source_dir, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Iterate over the images in the folder
        image_files = sorted(os.listdir(folder_path))
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            
            # Load the image
            with Image.open(image_path) as img:
                # Convert the image to a PyTorch tensor and add a batch dimension
                tensor = torch.tensor(np.array(img)).unsqueeze(0)
                tensor_list.append(tensor)

# Concatenate all tensors along the 0th dimension
all_tensors = torch.cat(tensor_list, dim=0)
print('The shape generated images tensors is:', all_tensors.shape)
# Save the combined tensor
combined_tensor_path = os.path.join(target_dir, "recon_images.pt")
torch.save(all_tensors, combined_tensor_path)

############################################################################################
# Define the source and target directories
source_dir = '/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images'
target_dir = f"./results/models/{args.encoder_type}/{args.val}/{sub}"
# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize a list to hold all the image tensors
tensor_list = []

# Iterate over the folders in the source directory
for folder_name in sorted(os.listdir(source_dir)):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Iterate over the images in the folder
        image_files = sorted(os.listdir(folder_path))
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            
            # Load the image
            with Image.open(image_path) as img:
                # Convert the image to a PyTorch tensor and add a batch dimension
                tensor = torch.tensor(np.array(img)).unsqueeze(0)
                tensor_list.append(tensor)

# Concatenate all tensors along the 0th dimension
all_tensors = torch.cat(tensor_list, dim=0)
print('The shape of real images is:', all_tensors.shape)
# Save the combined tensor
combined_tensor_path = os.path.join(target_dir, "real_images.pt")
torch.save(all_tensors, combined_tensor_path)

recon_path = f"./results/models/{args.encoder_type}/{args.val}/{sub}/recon_images.pt"
all_images_path = f"./results/models/{args.encoder_type}/{args.val}/{sub}/real_images.pt"
all_brain_recons = torch.load(f'{recon_path}', weights_only=True)
all_images = torch.load(f'{all_images_path}', weights_only=True)
all_brain_recons = all_brain_recons[::10]

all_images = all_images.permute(0, 3, 1, 2)
all_brain_recons = all_brain_recons.permute(0, 3, 1, 2)
print(all_images.shape)
print(all_brain_recons.shape)

all_images = all_images.to(device)
all_brain_recons = all_brain_recons.to(device)

######################################################################
fig, axs = plt.subplots(2, 20, figsize=(25, 10))

for i in range(20):
    img = all_images[i].detach()
    img = transforms.ToPILImage()(img/255.0)
    axs[0, i].imshow(np.asarray(img))
    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

for i in range(20):
    img = all_brain_recons[i].detach()
    # print(img)
    img = transforms.ToPILImage()(img/255.0)
    axs[1, i].imshow(np.asarray(img))
    axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

plt.show()
######################################################################
imsize = 256
all_images = transforms.Resize((imsize,imsize))(all_images)
all_brain_recons = transforms.Resize((imsize,imsize))(all_brain_recons)
# np.random.seed(0)
ind = np.flip(np.array([i for i in range(10)]))
# print(ind)

all_interleaved = torch.zeros(len(ind)*2,3,imsize,imsize).to(device)

icount = 0
for t in ind:
    all_interleaved[icount] = all_images[t].float().to(device)
    # print("all_interleaved", all_interleaved[0])
    all_interleaved[icount+1] = all_brain_recons[t].float().to(device)
    icount += 2

plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs,figsize):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.ToPILImage()(img/255.0)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
grid = make_grid(all_interleaved, nrow=10, padding=2)
# print(grid)
show(grid,figsize=(20,16))

print('--------------------------------------------------------------------------------')
print("Recons Minimum:", all_brain_recons.min().item())
print("Recons Maximum:", all_brain_recons.max().item())
print("Origin images Minimum:", all_images.min().item())
print("Origin images Maximum:", all_images.max().item())
####################################################################
# 2-Way Identification
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1


####################################################################
## PixCorr
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize((425, 425),interpolation=transforms.InterpolationMode.BILINEAR),
])

# Flatten images while keeping the batch dimension
all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
all_brain_recons_flattened = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()

print(all_images_flattened.shape)
print(all_brain_recons_flattened.shape)

corrsum = 0
for i in tqdm(range(len(all_images_flattened))):
    corrsum += np.corrcoef(all_images_flattened[i].numpy(), all_brain_recons_flattened[i].numpy())[0][1]
corrmean = corrsum / len(all_images_flattened)

pixcorr = corrmean
print('The PixCorr is:', pixcorr)

####################################################################
## SSIM
# see https://github.com/zijin-gu/meshconv-decoding/issues/3
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])

# convert image to grayscale with rgb2grey
img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
print("converted, now calculating ssim...")

ssim_score=[]
for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
    ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

ssim_num = np.mean(ssim_score)
print('The SSIM is:', ssim_num)


####################################################################
### AlexNet
from torchvision.models import alexnet, AlexNet_Weights
alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
alex_model.eval().requires_grad_(False)

# see alex_weights.transforms()
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# Ensure all_images and all_brain_recons are tensors on the correct device and are floating-point
all_images = all_images.to(device).float()  # Ensure conversion to float
all_brain_recons = all_brain_recons.to(device).float()  # Ensure conversion to float

layer = 'early, AlexNet(2)'
print(f"\n---{layer}---")
all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.4')
alexnet2 = np.mean(all_per_correct)
print(f"2-way Percent Correct alexnet2: {alexnet2:.4f}")

layer = 'mid, AlexNet(5)'
print(f"\n---{layer}---")
all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.11')
alexnet5 = np.mean(all_per_correct)
print(f"2-way Percent Correct alexnet5: {alexnet5:.4f}")

####################################################################
### InceptionV3
from torchvision.models import inception_v3, Inception_V3_Weights
weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                           return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images, inception_model, preprocess, 'avgpool')
        
inception = np.mean(all_per_correct)
print(f"2-way Percent Correct, InceptionV3: {inception:.4f}")

####################################################################
### CLIP
import clip
clip_model, preprocess = clip.load("ViT-L/14", device=device)

preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images,
                                        clip_model.encode_image, preprocess, None) # final layer
clip_ = np.mean(all_per_correct)
print(f"2-way Percent Correct, CLIP: {clip_:.4f}")
#############################################################################
### Efficient Net-B
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                    return_nodes=['avgpool']).to(device)
eff_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Process all_images
# gt = eff_model(torch.stack([preprocess(img.float()) for img in all_images.to(device)], dim=0))['avgpool']
gt = eff_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()

# Process all_brain_recons
# fake = eff_model(torch.stack([preprocess(recon.float()) for recon in all_brain_recons.to(device)], dim=0))['avgpool']
fake = eff_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("Distance, Efficient Net-B:", effnet)

####################################################################
### SwAV

swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
swav_model = create_feature_extractor(swav_model, 
                                    return_nodes=['avgpool']).to(device)
swav_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.float()/255.0),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gt = swav_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = swav_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("Distance, SwAV:",swav)

####################################################################
# Create a dictionary to store variable names and their corresponding values
data = {
    "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
    "Value": [pixcorr, ssim_num, alexnet2, alexnet5, inception, clip_, effnet, swav],
}

df = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7], data = [data["Metric"], [f"{num:.5f}" for num in data["Value"]]])
print(df.to_string(index=False))

df.to_csv(f'only_image_pred_mindeye_1.0.csv')

