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