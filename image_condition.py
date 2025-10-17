import os
from PIL import Image
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from model import Qformer, ATMS_classification_50, Proj_eeg_img, Proj_eeg_text, BrainDiffusionPrior, PriorNetwork
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
import argparse
import utils
from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from custom_pipeline import Generator4Embeds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
parser.add_argument('--result_path', default='./results/generation/' , type=str)
parser.add_argument('--test_image_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images' , type=str)
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
parser.add_argument('--in_dim', default=1024, type=int, help='the dimension of input')
parser.add_argument('--num_tokens', default=1, type=int, help='the number of text tokens')
parser.add_argument('--clip_dim', default=1024, type=int, help='the dimension of clip text embeddings')
parser.add_argument('--n_blocks', default=2, type=int, help='the number of blocks in BrainNetwork')
parser.add_argument('--depth', default=2, type=int, help='the depth in PriorNetwork')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--alpha', default=0.5, type=float, help='parameters for balancing EEG_image and EEG_text')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--model_type', default='ViT-H-14', type=str)
parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_class_alpha0.5', type=str)


class EEGToImageModule(nn.Module):
    def __init__(self):
        super(EEGToImageModule, self).__init__()
    def forward(self, x):
        return x

def get_eeg_data(args, sub):
    # load traindata and validation data
    n_classes = 1654
    samples_per_class = 10
    label_list = []
    file_name = 'preprocessed_eeg_training.npy'
    file_path = os.path.join(args.data_path, sub, file_name)
    data = np.load(file_path, allow_pickle=True)
    # （16540，4， 63， 250）——>（16540， 63， 250）
    preprocessed_eeg_data = data['preprocessed_eeg_data']
    preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)
    train_data = torch.from_numpy(preprocessed_eeg_data).float().detach()
    # train_data = train_data.view(-1, *train_data[0][0].shape)

    for i in range(n_classes):
        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
        label_list.append(labels)
    train_label = torch.cat(label_list, dim=0)
    # train_label = train_label.repeat_interleave(4)

    # load test data
    file_name = 'preprocessed_eeg_test.npy'
    file_path = os.path.join(args.data_path, sub, file_name)
    data = np.load(file_path, allow_pickle=True)
    # (200, 80, 63, 250) ——> (200, 63, 250)
    preprocessed_eeg_data = data['preprocessed_eeg_data']
    preprocessed_eeg_data = np.mean(preprocessed_eeg_data, axis=1)
    test_data = torch.from_numpy(preprocessed_eeg_data).float().detach()
    test_label = torch.from_numpy(np.arange(200)).detach()
    print('the shape of train_data is:', train_data.shape)
    print('the shape of train_label is:', train_label.shape)
    print('the shape of test_data is:', test_data.shape)
    print('the shape of test_label is:', test_label.shape)
    return train_data, train_label, test_data, test_label

def get_image_text_data(args):
    train_feature = torch.load(os.path.join(f'{args.model_type}_detail_class_features_train.pt'), weights_only=True)
    test_feature = torch.load(os.path.join(f'{args.model_type}_detail_class_features_test.pt'), weights_only=True)

    img_train_features = train_feature['img_features']
    img_test_features = test_feature['img_features']
    text_train_features = train_feature['text_features']
    text_test_features = test_feature['text_features']

    # text_train_features = text_train_features.unsqueeze(1).repeat(1, 10, 1)
    # text_train_features = text_train_features.reshape(-1, text_train_features.shape[2])
    print('the shape of img_train_features is:', img_train_features.shape)
    print('the shape of img_test_features is:', img_test_features.shape)
    print('The shape of text_train_features is:', text_train_features.shape)
    print('The shape of text_test_features is:', text_test_features.shape)
    # (16540, 1024), (200, 1024)
    return img_train_features, img_test_features, text_train_features, text_test_features

def get_eeg_features(train_loader, eeg_model, eeg_image_proj, eeg_text_proj):
    eeg_model.eval()
    eeg_image_proj.eval()
    eeg_text_proj.eval()
    eeg_img_features_list = []
    eeg_text_features_list = []
    with torch.no_grad():
        for batch_idx, (eeg_data, _) in enumerate(train_loader):
            eeg_data = eeg_data.to(device).float()
            eeg_features = eeg_model(eeg_data)
            eeg_img_features = eeg_image_proj(eeg_features)
            eeg_text_features = eeg_text_proj(eeg_features)
            
            eeg_img_features_list.append(eeg_img_features)
            eeg_text_features_list.append(eeg_text_features)

    eeg_img_features = torch.cat(eeg_img_features_list, dim=0).cpu()
    eeg_text_features = torch.cat(eeg_text_features_list, dim=0).cpu()
    return eeg_img_features, eeg_text_features

def main(): 
    args = parser.parse_args()
    subjects = ['sub-08']

    seed_n = args.seed
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    
    for sub in subjects:
        eeg_model = ATMS_classification_50()
        eeg_image_proj = Proj_eeg_img()
        eeg_text_proj = Proj_eeg_text()
        print('The number of parameters of eeg_model:', sum([p.numel() for p in eeg_model.parameters()]))
        print('The number of parameters of eeg_image_proj:', sum([p.numel() for p in eeg_image_proj.parameters()]))
        print('The number of parameters of eeg_text_proj:', sum([p.numel() for p in eeg_text_proj.parameters()]))
        eeg_model.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/eeg_model.pth", weights_only=True), strict=False)
        eeg_image_proj.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/eeg_img_proj.pth", weights_only=True), strict=False)
        eeg_text_proj.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/eeg_text_proj.pth", weights_only=True), strict=False)
        eeg_model = eeg_model.to(device)
        eeg_image_proj = eeg_image_proj.to(device)
        eeg_text_proj = eeg_text_proj.to(device)
        
        train_data, train_label, test_data, test_label = get_eeg_data(args, sub)
        img_train_features, img_test_features, text_train_features, text_test_features = get_image_text_data(args)
        # img_train_features = img_train_features.unsqueeze(1).repeat(1, 4, 1).view(-1, 1024)
        print('----------------------------------------------------------------------------------------------------------')
        print('The shape of train data is:', train_data.shape)
        print('The shape of img_train_features is:', img_train_features.shape)
        # get eeg test features
        eeg_model.eval()
        eeg_image_proj.eval()
        eeg_text_proj.eval()
        with torch.no_grad():
            test_data = test_data.to(device).float()
            eeg_test_features = eeg_model(test_data)
            eeg_img_test_features = eeg_image_proj(eeg_test_features)
            eeg_text_test_features = eeg_text_proj(eeg_test_features)
        eeg_img_test_features = eeg_img_test_features.cpu()
        eeg_text_test_features = eeg_text_test_features.cpu()
        # get eeg train features
        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        eeg_img_train_features, eeg_text_train_features = get_eeg_features(train_loader, eeg_model, eeg_image_proj, eeg_text_proj)
        
        print('The shape of eeg_img_train_features is:', eeg_img_train_features.shape)
        print('The shape of eeg_text_train_features is:', eeg_text_train_features.shape)
        print('The shape of eeg_img_test_features is:', eeg_img_test_features.shape)
        print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
        torch.save({
            'eeg_img_train_features': eeg_img_train_features,
            'eeg_text_train_features': eeg_text_train_features
        }, f"./data/ATM_S_eeg_train_features_{sub}.pt")
        torch.save({
            'eeg_img_test_features': eeg_img_test_features,
            'eeg_text_test_features': eeg_text_test_features
        }, f"./data/ATM_S_eeg_test_features_{sub}.pt")
        
        ######################################################################################################################
        # train image condition with mind-eye diffusion
        eeg_img_train_features = eeg_img_train_features.unsqueeze(1)
        img_train_features = img_train_features.unsqueeze(1)

        eeg_img_test_features = eeg_img_test_features.unsqueeze(1)
        img_test_features = img_test_features.unsqueeze(1)
        
        print('------------------------------------------------------------------------------------------')
        print('The shape of eeg_img_train_features is:', eeg_img_train_features.shape)
        print('The shape of img_train_features is:', img_train_features.shape)
        print('The shape of eeg_img_test_features is:', eeg_img_test_features.shape)
        print('The shape of img_test_features is:', img_test_features.shape)
        print('-----------------------------------------------------------------------------------------------------')
        print("EEG-to-image embeddings Minimum:", eeg_img_train_features.min().item())
        print("EEG-to-image embeddings Maximum:", eeg_img_train_features.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        print("Image embeddings Minimum:", img_train_features.min().item())
        print("Image embeddings Maximum:", img_train_features.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        # shuffle the training data
        train_shuffle = np.random.permutation(len(eeg_img_train_features))
        eeg_img_train_features = eeg_img_train_features[train_shuffle]
        img_train_features = img_train_features[train_shuffle]

        eeg_img_val_features = eeg_img_train_features[:740]
        img_val_features = img_train_features[:740]

        eeg_img_train_features = eeg_img_train_features[740:]
        img_train_features = img_train_features[740:]

        # Prepare data loaders
        train_dataset = TensorDataset(eeg_img_train_features, img_train_features)
        val_dataset = TensorDataset(eeg_img_val_features, img_val_features)
        test_dataset = TensorDataset(eeg_img_test_features, img_test_features)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        criterion_mse = nn.MSELoss().to(device)
        model = EEGToImageModule()
        # setup diffusion prior network
        timesteps = 100
        prior_network = PriorNetwork(
                dim=args.clip_dim,
                depth=args.depth,
                dim_head=256,
                heads=8,
                causal=False,
                num_tokens=args.num_tokens,
                learned_query_mode="pos_emb"
            ).to(device)

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=args.clip_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(device)

        print('The number of params in DiffusionPrior:')
        utils.count_params(model.diffusion_prior)
        print('The number of params in overall model:')
        utils.count_params(model)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_grouped_parameters = [
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.lr)
        total_steps=int(np.floor(args.num_epochs*len(train_loader)))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2/args.num_epochs
        )

        # Training loop
        best_val_loss = np.inf
        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss, _ = model.diffusion_prior(text_embed=inputs, image_embed=targets)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
            train_loss = train_loss / (batch_idx + 1)
            
            if (epoch + 1) % 1 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss, _ = model.diffusion_prior(text_embed=inputs, image_embed=targets)
                        val_loss += loss.item()
                    val_loss = val_loss / (idx + 1)
                print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth')
                    print("Model saved as image_condition.pth")

        # Testing loop
        model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth', weights_only=True), strict=False)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                prior_out = model.diffusion_prior.p_sample_loop(inputs.shape, text_cond=dict(text_embed=inputs), cond_scale=1.0, timesteps=20)
                loss = criterion_mse(prior_out, targets)
                test_loss += loss.item()
            test_loss = test_loss / (idx + 1)
        print(f"Test Loss: {test_loss}")

        image_model = EEGToImageModule()
        # setup diffusion prior network
        timesteps = 100
        prior_network = PriorNetwork(
                dim=args.clip_dim,
                depth=args.depth,
                dim_head=256,
                heads=8,
                causal=False,
                num_tokens=args.num_tokens,
                learned_query_mode="pos_emb"
            ).to(device)

        image_model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=args.clip_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(device)

        image_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth', weights_only=True), strict=False)
        # Testing loop
        image_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                prior_out = model.diffusion_prior.p_sample_loop(inputs.shape, text_cond=dict(text_embed=inputs), cond_scale=1.0, timesteps=20)
                loss = criterion_mse(prior_out, targets)
                test_loss += loss.item()
            test_loss = test_loss / (idx + 1)
        print(f"After loading, the test Loss: {test_loss}")
        
        #######################################################################################################################
        # # train image condition with Q-former
        # eeg_img_train_features = eeg_img_train_features.unsqueeze(1)
        # img_train_features = img_train_features.unsqueeze(1)

        # eeg_img_test_features = eeg_img_test_features.unsqueeze(1)
        # img_test_features = img_test_features.unsqueeze(1)
        
        # print('------------------------------------------------------------------------------------------')
        # print('The shape of eeg_img_train_features is:', eeg_img_train_features.shape)
        # print('The shape of img_train_features is:', img_train_features.shape)
        # print('The shape of eeg_img_test_features is:', eeg_img_test_features.shape)
        # print('The shape of img_test_features is:', img_test_features.shape)

        # # Instantiate the model, loss function, and optimizer
        # model = Qformer(input_emb_size=1024, emb_size=1024, num_query_token=1, depth=6, heads=8).to(torch.bfloat16).to(device)
        # criterion_mse = nn.MSELoss().to(device)
        # # criterion_KL = nn.KLDivLoss(reduction="batchmean").to(device)
        # optimizer = optim.AdamW(model.parameters(), lr=0.0002)

        # # shuffle the training data
        # train_shuffle = np.random.permutation(len(eeg_img_train_features))
        # eeg_img_train_features = eeg_img_train_features[train_shuffle]
        # img_train_features = img_train_features[train_shuffle]

        # eeg_img_val_features = eeg_img_train_features[:740]
        # img_val_features = img_train_features[:740]

        # eeg_img_train_features = eeg_img_train_features[740:]
        # img_train_features = img_train_features[740:]

        # # Prepare data loaders
        # train_dataset = TensorDataset(eeg_img_train_features, img_train_features)
        # val_dataset = TensorDataset(eeg_img_val_features, img_val_features)
        # test_dataset = TensorDataset(eeg_img_test_features, img_test_features)

        # train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # # Training loop
        # num_epochs = 100
        # best_val_loss = np.inf
        # for epoch in range(num_epochs):
        #     model.train()
        #     train_loss = 0.0
        #     for batch_idx, (inputs, targets) in enumerate(train_loader):
        #         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        #         optimizer.zero_grad()
        #         outputs = model(inputs)
        #         loss = criterion_mse(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        #         train_loss += loss.item()
        #     train_loss = train_loss / (batch_idx + 1)
            
        #     if (epoch + 1) % 1 == 0:
        #         model.eval()
        #         val_loss = 0.0
        #         with torch.no_grad():
        #             for idx, (inputs, targets) in enumerate(val_loader):
        #                 inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        #                 outputs = model(inputs)
        #                 loss = criterion_mse(outputs, targets)
        #                 val_loss += loss.item()
        #             val_loss = val_loss / (idx + 1)
        #         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        #         if val_loss < best_val_loss:
        #             best_val_loss = val_loss
        #             torch.save(model.state_dict(), f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth')
        #             print("Model saved as image_condition.pth")

        # # Testing loop
        # model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth', weights_only=True), strict=False)
        # model.eval()
        # test_loss = 0.0
        # with torch.no_grad():
        #     for idx, (inputs, targets) in enumerate(test_loader):
        #         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        #         outputs = model(inputs)
        #         loss = criterion_mse(outputs, targets)
        #         test_loss += loss.item()
        #     test_loss = test_loss / (idx + 1)
        # print(f"Test Loss: {test_loss}")

        # image_model = Qformer(input_emb_size=1024, emb_size=1024, num_query_token=1, depth=6, heads=8).to(torch.bfloat16).to(device)
        # image_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth', weights_only=True), strict=False)
        # # Testing loop
        # image_model.eval()
        # test_loss = 0.0
        # with torch.no_grad():
        #     for idx, (inputs, targets) in enumerate(test_loader):
        #         inputs, targets = inputs.to(torch.bfloat16).to(device), targets.to(torch.bfloat16).to(device)
        #         outputs = image_model(inputs)
        #         loss = criterion_mse(outputs, targets)
        #         test_loss += loss.item()
        #     test_loss = test_loss / (idx + 1)
        # print(f"After loading, the test Loss: {test_loss}")
        
        ##############################################################################################################################
        # # train the eeg model with diffusion prior
        # eeg_img_dataset = EmbeddingDataset(c_embeddings=eeg_img_train_features, h_embeddings=img_train_features)
        # eeg_img_dataloader = DataLoader(eeg_img_dataset, batch_size=256, shuffle=True, num_workers=64)

        # diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
        # # number of parameters
        # print('The parameters number of img_diffusion_prior is:', sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
        # # eeg-to-image
        # pipe = Pipe(diffusion_prior, device=device)
        # # load pretrained model
        # model_name = 'diffusion_prior_eeg_to_image'
        # # pipe.diffusion_prior.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/{model_name}.pt", map_location=device))
        # pipe.train(eeg_img_dataloader, num_epochs=150, learning_rate=1e-3) # to 0.142 
        # save_path = f"./results/models/{args.encoder_type}/{args.val}/{sub}/{model_name}.pt"
        # directory = os.path.dirname(save_path)
        # # # Create the directory if it doesn't exist
        # os.makedirs(directory, exist_ok=True)
        # torch.save(pipe.diffusion_prior.state_dict(), save_path)
        
if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))