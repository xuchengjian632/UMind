# # text condition with Qformer and diffusion prior
# import os
# import sys
# import json
# import argparse
# import numpy as np
# import math
# from einops import rearrange
# import time
# import random
# import string
# import h5py
# from tqdm import tqdm
# import webdataset as wds

# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import utils
# from model import PriorNetwork, BrainDiffusionPrior, BrainNetwork
# from model import Qformer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
# parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
# parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
# parser.add_argument('--in_dim', default=1024, type=int, help='the dimension of input')
# parser.add_argument('--num_tokens', default=1, type=int, help='the number of text tokens')
# parser.add_argument('--clip_dim', default=1280, type=int, help='the dimension of clip text embeddings')
# parser.add_argument('--n_blocks', default=2, type=int, help='the number of blocks in BrainNetwork')
# parser.add_argument('--depth', default=4, type=int, help='the depth in PriorNetwork')
# parser.add_argument('--batch_size', default=256, type=int, help='batch size')
# parser.add_argument('--num_epochs', default=100, type=int, help='the number of epochs')
# parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
# parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)
# args = parser.parse_args()

# seed_n = args.seed
# # seed_n = np.random.randint(args.seed)
# print('seed is ' + str(seed_n))
# random.seed(seed_n)
# np.random.seed(seed_n)
# torch.manual_seed(seed_n)
# torch.cuda.manual_seed(seed_n)
# torch.cuda.manual_seed_all(seed_n)  # if using multi-GPU.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# sub = 'sub-08'

# eeg_text_train_features = torch.load(os.path.join(f'./data/ATM_S_eeg_train_features_{sub}.pt'), weights_only=True)['eeg_text_train_features']
# eeg_text_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_text_test_features']

# train_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_train.pt'), weights_only=True)['pooled_prompt_embeds']
# test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']
# # train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1).repeat(1, 4, 1).view(-1, 1280)
# eeg_text_train_features = eeg_text_train_features.unsqueeze(1)
# eeg_text_test_features = eeg_text_test_features.unsqueeze(1)

# train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1)
# test_pooled_prompt_embeds = test_pooled_prompt_embeds.unsqueeze(1)

# print('-----------------------------------------------------------------------------------------------------')
# print("EEG-to-text embeddings Minimum:", eeg_text_train_features.min().item())
# print("EEG-to-text embeddings Maximum:", eeg_text_train_features.max().item())
# print('-----------------------------------------------------------------------------------------------------')
# print("Pooled Prompt embeddings Minimum:", train_pooled_prompt_embeds.min().item())
# print("Pooled Prompt embeddings Maximum:", train_pooled_prompt_embeds.max().item())
# print('-----------------------------------------------------------------------------------------------------')
# print('The shape of eeg_text_train_features is:', eeg_text_train_features.shape)
# print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
# print('The shape of train_pooled_prompt_embeds is:', train_pooled_prompt_embeds.shape)
# print('The shape of test_pooled_prompt_embeds is:', test_pooled_prompt_embeds.shape)
# print('-----------------------------------------------------------------------------------------------------')
# # shuffle the training data
# train_shuffle = np.random.permutation(len(eeg_text_train_features))
# eeg_text_train_features = eeg_text_train_features[train_shuffle]
# train_pooled_prompt_embeds = train_pooled_prompt_embeds[train_shuffle]

# eeg_text_val_features = eeg_text_train_features[:740]
# val_pooled_prompt_embeds = train_pooled_prompt_embeds[:740]

# eeg_text_train_features = eeg_text_train_features[740:]
# train_pooled_prompt_embeds = train_pooled_prompt_embeds[740:]

# # Prepare data loaders
# train_dataset = torch.utils.data.TensorDataset(eeg_text_train_features, train_pooled_prompt_embeds)
# val_dataset = torch.utils.data.TensorDataset(eeg_text_val_features, val_pooled_prompt_embeds)
# test_dataset = torch.utils.data.TensorDataset(eeg_text_test_features, test_pooled_prompt_embeds)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# class EEGToImageModule(nn.Module):
#     def __init__(self):
#         super(EEGToImageModule, self).__init__()
#     def forward(self, x):
#         return x
        
# model = EEGToImageModule()
# model.backbone = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=1, heads=5).to(torch.bfloat16).to(device)
# # model.backbone = BrainNetwork(
# #     in_dim=args.in_dim,
# #     out_dim=args.num_tokens*args.clip_dim,
# #     seq_len=1,
# #     n_blocks=args.n_blocks,
# #     clip_size=args.clip_dim
# # ).to(torch.bfloat16).to(device)
# print('The number of params in BrainNetwork:')
# utils.count_params(model.backbone)

# # setup diffusion prior network
# timesteps = 100
# prior_network = PriorNetwork(
#         dim=args.clip_dim,
#         depth=args.depth,
#         dim_head=256,
#         heads=8,
#         causal=False,
#         num_tokens=args.num_tokens,
#         learned_query_mode="pos_emb"
#     ).to(torch.bfloat16).to(device)

# model.diffusion_prior = BrainDiffusionPrior(
#     net=prior_network,
#     image_embed_dim=args.clip_dim,
#     condition_on_text_encodings=False,
#     timesteps=timesteps,
#     cond_drop_prob=0.2,
#     image_embed_scale=None,
# ).to(torch.bfloat16).to(device)

# print('The number of params in DiffusionPrior:')
# utils.count_params(model.diffusion_prior)
# print('The number of params in overall model:')
# utils.count_params(model)

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# opt_grouped_parameters = [
#     {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
#     {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
# ]
# opt_grouped_parameters.extend([
#     {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
#     {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ])

# optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.lr)

# criterion_mse = nn.MSELoss()
# criterion_l1 = nn.L1Loss()
# total_steps=int(np.floor(args.num_epochs*len(train_loader)))
# print("total_steps", total_steps)
# lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=args.lr,
#     total_steps=total_steps,
#     final_div_factor=1000,
#     last_epoch=-1,
#     pct_start=2/args.num_epochs
# )

# # Training loop
# best_val_loss = np.inf
# for epoch in range(args.num_epochs):
#     model.train()
#     train_loss = 0.0
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         optimizer.zero_grad()
#         backbone = model.backbone(inputs)
#         loss, _ = model.diffusion_prior(text_embed=backbone, image_embed=targets)
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         train_loss += loss.item()
#     train_loss = train_loss / (batch_idx + 1)
    
#     if (epoch + 1) % 1 == 0:
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for idx, (inputs, targets) in enumerate(val_loader):
#                 inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#                 backbone = model.backbone(inputs)
#                 loss, _ = model.diffusion_prior(text_embed=backbone, image_embed=targets)
#                 val_loss += loss.item()
#             val_loss = val_loss / (idx + 1)
#         print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth')
#             print("Model saved as text_pool_condition.pth")

# # Testing loop
# model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for idx, (inputs, targets) in enumerate(test_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         backbone = model.backbone(inputs)
#         prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, text_cond=dict(text_embed=backbone), cond_scale=1.0, timesteps=20)
#         loss = criterion_mse(prior_out, targets)
#         test_loss += loss.item()
#     test_loss = test_loss / (idx + 1)
# print(f"Test Loss: {test_loss}")

# text_pool_model = EEGToImageModule()
# text_pool_model.backbone = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
# # model.backbone = BrainNetwork(
# #     in_dim=args.in_dim,
# #     out_dim=args.num_tokens*args.clip_dim,
# #     seq_len=1,
# #     n_blocks=args.n_blocks,
# #     clip_size=args.clip_dim
# # ).to(torch.bfloat16).to(device)
# # setup diffusion prior network
# timesteps = 100
# prior_network = PriorNetwork(
#         dim=args.clip_dim,
#         depth=args.depth,
#         dim_head=256,
#         heads=8,
#         causal=False,
#         num_tokens=args.num_tokens,
#         learned_query_mode="pos_emb"
#     ).to(torch.bfloat16).to(device)

# text_pool_model.diffusion_prior = BrainDiffusionPrior(
#     net=prior_network,
#     image_embed_dim=args.clip_dim,
#     condition_on_text_encodings=False,
#     timesteps=timesteps,
#     cond_drop_prob=0.2,
#     image_embed_scale=None,
# ).to(torch.bfloat16).to(device)

# text_pool_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
# # Testing loop
# text_pool_model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for idx, (inputs, targets) in enumerate(test_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         backbone = text_pool_model.backbone(inputs)
#         prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, text_cond=dict(text_embed=backbone), cond_scale=1.0, timesteps=20)
#         loss = criterion_mse(prior_out, targets)
#         test_loss += loss.item()
#     test_loss = test_loss / (idx + 1)
# print(f"After loading, the test Loss: {test_loss}")




##################################################################################################################################################
# # # use adversarial training to train Q-former
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from einops.layers.torch import Rearrange, Reduce
# import os
# import random
# import argparse
# from model import Qformer
# from model import Discriminator_pool_prompt
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
# parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
# parser.add_argument('--result_path', default='./results/generation/' , type=str)
# parser.add_argument('--test_image_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images' , type=str)
# parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
# parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
# parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
# parser.add_argument("--b1", default=0.5, type=float, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", default=0.999, type=float, help="adam: decay of first order momentum of gradient")
# parser.add_argument('--model_type', default='ViT-H-14', type=str)
# parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
# parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)
# args = parser.parse_args()

# seed_n = args.seed
# # seed_n = np.random.randint(args.seed)
# print('seed is ' + str(seed_n))
# random.seed(seed_n)
# np.random.seed(seed_n)
# torch.manual_seed(seed_n)
# torch.cuda.manual_seed(seed_n)
# torch.cuda.manual_seed_all(seed_n)  # if using multi-GPU.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# sub = 'sub-08'

# eeg_text_train_features = torch.load(os.path.join(f'./data/ATM_S_eeg_train_features_{sub}.pt'), weights_only=True)['eeg_text_train_features']
# eeg_text_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_text_test_features']

# train_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_train.pt'), weights_only=True)['pooled_prompt_embeds']
# test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']
# # train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1).repeat(1, 4, 1).view(-1, 1280)
# eeg_text_train_features = eeg_text_train_features.unsqueeze(1)
# eeg_text_test_features = eeg_text_test_features.unsqueeze(1)

# train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1)
# test_pooled_prompt_embeds = test_pooled_prompt_embeds.unsqueeze(1)

# print('The shape of eeg_text_train_features is:', eeg_text_train_features.shape)
# print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
# print('The shape of train_pooled_prompt_embeds is:', train_pooled_prompt_embeds.shape)
# print('The shape of test_pooled_prompt_embeds is:', test_pooled_prompt_embeds.shape)

# # shuffle the training data
# train_shuffle = np.random.permutation(len(eeg_text_train_features))
# eeg_text_train_features = eeg_text_train_features[train_shuffle]
# train_pooled_prompt_embeds = train_pooled_prompt_embeds[train_shuffle]

# eeg_text_val_features = eeg_text_train_features[:740]
# val_pooled_prompt_embeds = train_pooled_prompt_embeds[:740]

# eeg_text_train_features = eeg_text_train_features[740:]
# train_pooled_prompt_embeds = train_pooled_prompt_embeds[740:]

# # Prepare data loaders
# train_dataset = TensorDataset(eeg_text_train_features, train_pooled_prompt_embeds)
# val_dataset = TensorDataset(eeg_text_val_features, val_pooled_prompt_embeds)
# test_dataset = TensorDataset(eeg_text_test_features, test_pooled_prompt_embeds)

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# # Instantiate the model, loss function, and optimizer
# generator = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=4, heads=10).to(torch.bfloat16).to(device)
# discriminator = Discriminator_pool_prompt().to(torch.bfloat16).to(device)
# print('The number of parameters of generator:', sum([p.numel() for p in generator.parameters()]))
# print('The number of parameters of discriminator:', sum([p.numel() for p in discriminator.parameters()]))
# adversarial_loss = torch.nn.BCELoss().to(device)
# criterion_mse = nn.MSELoss().to(device)
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# # Training loop
# num_epochs = 20
# best_val_loss = np.inf
# for epoch in range(num_epochs):
#     generator.train()
#     discriminator.train()
#     train_loss = 0.0
#     train_g_loss = 0.0
#     train_d_loss = 0.0
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         # Adversarial ground truths
#         valid = torch.ones((inputs.size(0), 1)).to(torch.bfloat16).to(device)
#         fake = torch.zeros((inputs.size(0), 1)).to(torch.bfloat16).to(device)
#         # -----------------
#         #  Train Generator
#         # -----------------
#         optimizer_G.zero_grad()
#         outputs = generator(inputs)
#         g_loss = adversarial_loss(discriminator(outputs), valid) + criterion_mse(outputs, targets)
#         g_loss.backward()
#         optimizer_G.step()
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#         optimizer_D.zero_grad()
#         real_loss = adversarial_loss(discriminator(targets), valid)
#         fake_loss = adversarial_loss(discriminator(outputs.detach()), fake)
#         d_loss = (real_loss + fake_loss) / 2
        
#         d_loss.backward()
#         optimizer_D.step()
        
#         train_loss += g_loss.item() + d_loss.item()
#         train_g_loss += g_loss.item()
#         train_d_loss += d_loss.item()
#     train_loss = train_loss / (batch_idx + 1)
#     train_g_loss = train_g_loss / (batch_idx + 1)
#     train_d_loss = train_d_loss / (batch_idx + 1)
    
#     if (epoch + 1) % 1 == 0:
#         generator.eval()
#         discriminator.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for idx, (inputs, targets) in enumerate(val_loader):
#                 inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#                 outputs = generator(inputs)
#                 loss = criterion_mse(outputs, targets)
#                 val_loss += loss.item()
#                 # # Adversarial ground truths
#                 # valid = torch.ones((inputs.size(0), 1)).to(torch.bfloat16).to(device)
#                 # fake = torch.zeros((inputs.size(0), 1)).to(torch.bfloat16).to(device)
#                 # #  Generator
#                 # outputs = generator(inputs)
#                 # g_loss = adversarial_loss(discriminator(outputs), valid) + criterion_mse(outputs, targets)
#                 # #  Discriminator
#                 # real_loss = adversarial_loss(discriminator(targets), valid)
#                 # fake_loss = adversarial_loss(discriminator(outputs), fake)
#                 # d_loss = (real_loss + fake_loss) / 2
#                 # val_loss += g_loss.item() + d_loss.item()
#             val_loss = val_loss / (idx + 1)
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train generator Loss: {train_g_loss}, Train discriminator Loss: {train_d_loss}, Validation Loss: {val_loss}")
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(generator.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth')
#             print("Model saved as text_pool_condition.pth")

# # Testing loop
# generator.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
# generator.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for idx, (inputs, targets) in enumerate(test_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         outputs = generator(inputs)
#         loss = criterion_mse(outputs, targets)
#         test_loss += loss.item()
#     test_loss = test_loss / (idx + 1)
# print(f"Test Loss: {test_loss}")

# text_pool_model = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=4, heads=10).to(torch.bfloat16).to(device)
# text_pool_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
# # Testing loop
# text_pool_model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for idx, (inputs, targets) in enumerate(test_loader):
#         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
#         outputs = text_pool_model(inputs)
#         loss = criterion_mse(outputs, targets)
#         test_loss += loss.item()
#     test_loss = test_loss / (idx + 1)
# print(f"After loading, the test Loss: {test_loss}")


# # ###############################################################################################################################
# only use mse loss to train Q-former
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from einops.layers.torch import Rearrange, Reduce
import os
import random
import argparse
from model import Qformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
parser.add_argument('--result_path', default='./results/generation/' , type=str)
parser.add_argument('--test_image_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images' , type=str)
parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--model_type', default='ViT-H-14', type=str)
parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_class_alpha0.5', type=str)
args = parser.parse_args()

seed_n = args.seed
# seed_n = np.random.randint(args.seed)
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)  # if using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sub = 'sub-08'
# Define the neural network
# class PoolPromptProjector(nn.Sequential):
#     def __init__(self, proj_dim=1280):
#         super().__init__(
#             nn.Linear(1024, 1024),
#             # nn.ELU(),
#             nn.LayerNorm(1024),
#             nn.Linear(1024, 1280),
#             # nn.ELU(),
#             nn.LayerNorm(proj_dim),
#             )

eeg_text_train_features = torch.load(os.path.join(f'./data/ATM_S_eeg_train_features_{sub}.pt'), weights_only=True)['eeg_text_train_features']
eeg_text_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_text_test_features']

# train_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_train.pt'), weights_only=True)['pooled_prompt_embeds']
# test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']

# train_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_only_text_embeds_train.pt'), weights_only=True)['pooled_prompt_embeds']
# test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_only_text_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']

train_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_only_class_embeds_train.pt'), weights_only=True)['pooled_prompt_embeds']
test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_only_class_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']
# train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1).repeat(1, 4, 1).view(-1, 1280)
eeg_text_train_features = eeg_text_train_features.unsqueeze(1)
eeg_text_test_features = eeg_text_test_features.unsqueeze(1)

train_pooled_prompt_embeds = train_pooled_prompt_embeds.unsqueeze(1)
test_pooled_prompt_embeds = test_pooled_prompt_embeds.unsqueeze(1)

print('-----------------------------------------------------------------------------------------------------')
print("EEG-to-text embeddings Minimum:", eeg_text_train_features.min().item())
print("EEG-to-text embeddings Maximum:", eeg_text_train_features.max().item())
print('-----------------------------------------------------------------------------------------------------')
print("Pooled Prompt embeddings Minimum:", train_pooled_prompt_embeds.min().item())
print("Pooled Prompt embeddings Maximum:", train_pooled_prompt_embeds.max().item())
print('-----------------------------------------------------------------------------------------------------')
print('The shape of eeg_text_train_features is:', eeg_text_train_features.shape)
print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
print('The shape of train_pooled_prompt_embeds is:', train_pooled_prompt_embeds.shape)
print('The shape of test_pooled_prompt_embeds is:', test_pooled_prompt_embeds.shape)
print('-----------------------------------------------------------------------------------------------------')
# Instantiate the model, loss function, and optimizer
model = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
criterion_mse = nn.MSELoss().to(device)
# criterion_KL = nn.KLDivLoss(reduction="batchmean").to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0002)

# shuffle the training data
train_shuffle = np.random.permutation(len(eeg_text_train_features))
eeg_text_train_features = eeg_text_train_features[train_shuffle]
train_pooled_prompt_embeds = train_pooled_prompt_embeds[train_shuffle]

eeg_text_val_features = eeg_text_train_features[:740]
val_pooled_prompt_embeds = train_pooled_prompt_embeds[:740]

eeg_text_train_features = eeg_text_train_features[740:]
train_pooled_prompt_embeds = train_pooled_prompt_embeds[740:]

# Prepare data loaders
train_dataset = TensorDataset(eeg_text_train_features, train_pooled_prompt_embeds)
val_dataset = TensorDataset(eeg_text_val_features, val_pooled_prompt_embeds)
test_dataset = TensorDataset(eeg_text_test_features, test_pooled_prompt_embeds)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Training loop
num_epochs = 100
best_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion_mse(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / (batch_idx + 1)
    
    if (epoch + 1) % 1 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
                outputs = model(inputs)
                loss = criterion_mse(outputs, targets)
                val_loss += loss.item()
            val_loss = val_loss / (idx + 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth')
            print("Model saved as text_pool_condition.pth")

# Testing loop
model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
model.eval()
test_loss = 0.0
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        outputs = model(inputs)
        loss = criterion_mse(outputs, targets)
        test_loss += loss.item()
    test_loss = test_loss / (idx + 1)
print(f"Test Loss: {test_loss}")

text_pool_model = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
text_pool_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
# Testing loop
text_pool_model.eval()
test_loss = 0.0
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        outputs = text_pool_model(inputs)
        loss = criterion_mse(outputs, targets)
        test_loss += loss.item()
    test_loss = test_loss / (idx + 1)
print(f"After loading, the test Loss: {test_loss}")

# Save the trained model
# torch.save(model.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth')
# print("Model saved as text_pool_condition.pth")