# images generation with Qformer and diffusion prior for text prediction
import os
from PIL import Image
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from model import Qformer
from model import EEG_embeds_to_vae_latent
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
import argparse
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from custom_pipeline import Generator4Embeds
from custom_pipeline_sd15 import Generator4Embeds_sd15
from custom_pipeline_low_level import Generator4Embeds_latent2img
from model import PriorNetwork, BrainDiffusionPrior, BrainNetwork, BrainDiffusionPrior, PriorNetwork
import utils

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
parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--alpha', default=0.5, type=float, help='parameters for balancing EEG_image and EEG_text')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--model_type', default='ViT-H-14', type=str)
parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)

class EEGToImageModule(nn.Module):
    def __init__(self):
        super(EEGToImageModule, self).__init__()
    def forward(self, x):
        return x

def load_category(args):
    directory = args.test_image_path
    dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    dirnames.sort()
    texts = []
    for dir in dirnames:
        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue   
        texts.append(description)
    # texts = [text.replace('_', ' ') for text in texts]
    return texts

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

class EEGToImageModule(nn.Module):
    def __init__(self):
        super(EEGToImageModule, self).__init__()
    def forward(self, x):
        return x

def main(): 
    args = parser.parse_args()
    # Set seed value
    seed_n = args.seed
    # seed_n = np.random.randint(args.seed)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_n)

    num_sub = 1
    sub = 'sub-08'
    categories = load_category(args)
    eeg_text_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_text_test_features']
    eeg_img_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_img_test_features']
    
    print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
    print('The shape of eeg_img_test_features is:', eeg_img_test_features.shape)

    test_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_test.pt'), weights_only=True)['prompt_embeds']
    test_pooled_prompt_embeds = torch.load(os.path.join(f'./data/SDXL-text-encoder_prompt_embeds_test.pt'), weights_only=True)['pooled_prompt_embeds']
    
    img_train_features, img_test_features, text_train_features, text_test_features = get_image_text_data(args)
    
    
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
    ##############################################################################################################################
    # diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    # # number of parameters
    # print('The parameters number of diffusion_prior is:', sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    # # eeg-to-image
    # pipe = Pipe(diffusion_prior, device=device)
    # # load pretrained model
    # model_name = 'diffusion_prior_eeg_to_image'
    # pipe.diffusion_prior.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/{model_name}.pt", map_location=device, weights_only=True))
    ###############################################################################################################################
    # vae_model = EEG_embeds_to_vae_latent().to(torch.bfloat16).to(device)
    # vae_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/vae_condition.pth', weights_only=True), strict=False)
    
    # text_model = EEGToImageModule()
    # text_model.backbone = Qformer(input_emb_size=1024, emb_size=2048, num_query_token=77, depth=2).to(torch.bfloat16).to(device)
    # # setup diffusion prior network
    # timesteps = 100
    # prior_network = PriorNetwork(
    #         dim=2048,
    #         depth=2,
    #         dim_head=256,
    #         heads=8,
    #         causal=False,
    #         num_tokens=77,
    #         learned_query_mode="pos_emb"
    #     ).to(torch.bfloat16).to(device)

    # text_model.diffusion_prior = BrainDiffusionPrior(
    #     net=prior_network,
    #     image_embed_dim=2048,
    #     condition_on_text_encodings=False,
    #     timesteps=timesteps,
    #     cond_drop_prob=0.2,
    #     image_embed_scale=None,
    # ).to(torch.bfloat16).to(device)
    text_model = Qformer(input_emb_size=1024, emb_size=2048, num_query_token=77, depth=2).to(torch.bfloat16).to(device)
    text_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_condition.pth', weights_only=True), strict=False)
    
    # pool_text_model = EEGToImageModule()
    # pool_text_model.backbone = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
    # timesteps = 100
    # prior_network = PriorNetwork(
    #         dim=1280,
    #         depth=2,
    #         dim_head=256,
    #         heads=8,
    #         causal=False,
    #         num_tokens=1,
    #         learned_query_mode="pos_emb"
    #     ).to(torch.bfloat16).to(device)

    # pool_text_model.diffusion_prior = BrainDiffusionPrior(
    #     net=prior_network,
    #     image_embed_dim=1280,
    #     condition_on_text_encodings=False,
    #     timesteps=timesteps,
    #     cond_drop_prob=0.2,
    #     image_embed_scale=None,
    # ).to(torch.bfloat16).to(device)
    pool_text_model = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
    pool_text_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
    
    # vae_model.eval()
    image_model.eval()
    text_model.eval()
    pool_text_model.eval()
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    # generator = Generator4Embeds_latent2img(num_inference_steps=4, img2img_strength=0.75, device=device)
    directory = f"./results/generated_imgs/{sub}/only_text_pred_Qformer_1.0"
    with torch.no_grad():
        for k in range(200):
            eeg_img_embeds = eeg_img_test_features[k:k+1]
            eeg_text_embeds = eeg_text_test_features[k:k+1]
            
            prior_in = eeg_img_embeds.unsqueeze(0).to(device)
            prior_out = image_model.diffusion_prior.p_sample_loop(prior_in.shape, text_cond=dict(text_embed=prior_in), cond_scale=1.0, timesteps=20)
            image_embeds_pred = prior_out.squeeze(0)

            # vae_latent_pred = vae_model(eeg_img_embeds.unsqueeze(0).to(torch.bfloat16).to(device))
            
            prompt_embeds_pred = text_model(eeg_text_embeds.unsqueeze(0).to(torch.bfloat16).to(device))
            pool_prompt_embeds_pred = pool_text_model(eeg_text_embeds.unsqueeze(0).to(torch.bfloat16).to(device)).squeeze(0)
            print('The shape of image_embeds_pred is:', image_embeds_pred.shape)
            # print('The shape of vae_latent_pred is:', vae_latent_pred.shape)
            print('The shape of prompt_embeds_pred is:', prompt_embeds_pred.shape)
            print('The shape of pool_prompt_embeds_pred is:', pool_prompt_embeds_pred.shape)
            for j in range(10):
                # image = generator.generate(image_embeds_pred.to(dtype=torch.float16), generator=gen)
                # image = generator.generate(image_embeds_pred.to(dtype=torch.float16), prompt_embeds_pred, pool_prompt_embeds_pred, generator=gen)
                image = generator.generate(None, prompt_embeds_pred, pool_prompt_embeds_pred, generator=gen)
                # The shape of generated images is:(512, 512)
                # Construct the save path for each image
                path = f'{directory}/{categories[k]}/{j}.png'
                # Ensure the directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Save the PIL Image
                image.save(path)
                print(f'Image saved to {path}')

if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))

##########################################################################################################################
# # images generation with seperated models
# import os
# from PIL import Image
# import torch
# import torch.optim as optim
# from torch.nn import CrossEntropyLoss
# from torch.nn import functional as F
# from torch.optim import Adam
# from torch.utils.data import DataLoader
# import time
# import clip
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn as nn
# import torchvision.transforms as transforms
# import tqdm
# from model import Qformer
# from model import EEG_embeds_to_vae_latent
# from einops.layers.torch import Rearrange, Reduce
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import DataLoader, Dataset
# import random
# import argparse
# from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
# from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
# from custom_pipeline import Generator4Embeds
# from custom_pipeline_sd15 import Generator4Embeds_sd15
# from custom_pipeline_low_level import Generator4Embeds_latent2img, Generator4Embeds_img2img
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
# parser.add_argument('--data_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/Preprocessed_data_250Hz', type=str)
# parser.add_argument('--result_path', default='./results/generation/' , type=str)
# parser.add_argument('--test_image_path', default='/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images' , type=str)
# parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
# parser.add_argument('--epochs', default=50, type=int)
# parser.add_argument('--num_sub', default=10,type=int, help='the number of subjects used in the experiments')
# parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--alpha', default=0.5, type=float, help='parameters for balancing EEG_image and EEG_text')
# parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
# parser.add_argument('--model_type', default='ViT-H-14', type=str)
# parser.add_argument('--encoder_type', default='ATMS_classification_50', type=str)
# parser.add_argument('--val', default='val_acc_retrieval_classification_eeg_to_img_and_text_class_alpha0.5_both_mse2', type=str)

# def load_category(args):
#     directory = args.test_image_path
#     dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
#     dirnames.sort()
#     texts = []
#     for dir in dirnames:
#         try:
#             idx = dir.index('_')
#             description = dir[idx+1:]
#         except ValueError:
#             print(f"Skipped: {dir} due to no '_' found.")
#             continue   
#         texts.append(description)
#     # texts = [text.replace('_', ' ') for text in texts]
#     return texts

# def get_image_text_data(args):
#     train_feature = torch.load(os.path.join(f'{args.model_type}_detail_class_features_train.pt'), weights_only=True)
#     test_feature = torch.load(os.path.join(f'{args.model_type}_detail_class_features_test.pt'), weights_only=True)

#     img_train_features = train_feature['img_features']
#     img_test_features = test_feature['img_features']
#     text_train_features = train_feature['text_features']
#     text_test_features = test_feature['text_features']

#     # text_train_features = text_train_features.unsqueeze(1).repeat(1, 10, 1)
#     # text_train_features = text_train_features.reshape(-1, text_train_features.shape[2])
#     print('the shape of img_train_features is:', img_train_features.shape)
#     print('the shape of img_test_features is:', img_test_features.shape)
#     print('The shape of text_train_features is:', text_train_features.shape)
#     print('The shape of text_test_features is:', text_test_features.shape)
#     # (16540, 1024), (200, 1024)
#     return img_train_features, img_test_features, text_train_features, text_test_features

# def main(): 
#     args = parser.parse_args()
#     # Set seed value
#     seed_n = args.seed
#     print('seed is ' + str(seed_n))
#     random.seed(seed_n)
#     np.random.seed(seed_n)
#     torch.manual_seed(seed_n)
#     torch.cuda.manual_seed(seed_n)
#     torch.cuda.manual_seed_all(seed_n)
#     gen = torch.Generator(device=device)
#     gen.manual_seed(seed_n)

#     num_sub = 1
#     sub = 'sub-08'
#     categories = load_category(args)
#     eeg_text_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_text_test_features']
#     eeg_img_test_features = torch.load(os.path.join(f'./data/ATM_S_eeg_test_features_{sub}.pt'), weights_only=True)['eeg_img_test_features']
    
#     print('The shape of eeg_text_test_features is:', eeg_text_test_features.shape)
#     print('The shape of eeg_img_test_features is:', eeg_img_test_features.shape)
    
#     low_level_imgs = []
#     clip_preprocess = CLIPImageProcessor(
#             size={"height": 512, "width": 512},
#             crop_size={"height": 512, "width": 512},
#         )
#     low_level_img_path = f"./results/generated_imgs/low_level/{args.encoder_type}/{args.val}/{sub}"
#     image_files = sorted(os.listdir(low_level_img_path))
#     for image_name in image_files:
#         image_path = os.path.join(low_level_img_path, image_name)
#         low_level_image = Image.open(image_path)
#         low_level_image = clip_preprocess(low_level_image, return_tensors="pt").pixel_values
#         low_level_imgs.append(low_level_image)
#     low_level_imgs = torch.cat(low_level_imgs, dim=0)
#     print('The shape of low_level_imgs is:', low_level_imgs.shape)
    
#     img_train_features, img_test_features, text_train_features, text_test_features = get_image_text_data(args)
    
#     diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
#     # number of parameters
#     print('The parameters number of diffusion_prior is:', sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
#     # eeg-to-image
#     pipe = Pipe(diffusion_prior, device=device)
#     # load pretrained model
#     model_name = 'diffusion_prior_eeg_to_image'
#     pipe.diffusion_prior.load_state_dict(torch.load(f"./results/models/{args.encoder_type}/{args.val}/{sub}/{model_name}.pt", map_location=device, weights_only=True))
    
#     # vae_model = EEG_embeds_to_vae_latent().to(torch.bfloat16).to(device)
#     # vae_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/vae_condition.pth', weights_only=True), strict=False)
#     # image_model = Qformer(input_emb_size=1024, emb_size=1024, num_query_token=1, depth=6, heads=8).to(torch.bfloat16).to(device)
#     # image_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/image_condition.pth', weights_only=True), strict=False)
#     text_model = Qformer(input_emb_size=1024, emb_size=2048, num_query_token=77, depth=2).to(torch.bfloat16).to(device)
#     text_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_condition.pth', weights_only=True), strict=False)
#     pool_text_model = Qformer(input_emb_size=1024, emb_size=1280, num_query_token=1, depth=2, heads=5).to(torch.bfloat16).to(device)
#     pool_text_model.load_state_dict(torch.load(f'./results/models/{args.encoder_type}/{args.val}/{sub}/text_pool_condition.pth', weights_only=True), strict=False)
    
#     # vae_model.eval()
#     # image_model.eval()
#     text_model.eval()
#     pool_text_model.eval()
#     generator = Generator4Embeds(num_inference_steps=10, device=device)
#     # generator = Generator4Embeds_latent2img(num_inference_steps=4, img2img_strength=0.5, device=device)
#     # generator = Generator4Embeds_img2img(num_inference_steps=4, img2img_strength=0.75, device=device)
#     directory = f"./results/generated_imgs/{sub}/image_pred_text_pred_new_steps10_1.0"
#     with torch.no_grad():
#         for k in range(200):
#             eeg_img_embeds = eeg_img_test_features[k:k+1]
#             eeg_text_embeds = eeg_text_test_features[k:k+1]
#             # low_level_img = low_level_imgs[k:k+1]
#             image_embeds_pred = pipe.generate(c_embeds=eeg_img_embeds, num_inference_steps=50, guidance_scale=5.0)
#             # vae_latent_pred = vae_model(eeg_img_embeds.unsqueeze(0).to(torch.bfloat16).to(device))
#             prompt_embeds_pred = text_model(eeg_text_embeds.unsqueeze(0).to(torch.bfloat16).to(device))
#             pool_prompt_embeds_pred = pool_text_model(eeg_text_embeds.unsqueeze(0).to(torch.bfloat16).to(device)).squeeze(0)
#             print('The shape of image_embeds_pred is:', image_embeds_pred.shape)
#             # print('The shape of vae_latent_pred is:', vae_latent_pred.shape)
#             # print('The shape of low_level_img is:', low_level_img.shape)
#             print('The shape of prompt_embeds_pred is:', prompt_embeds_pred.shape)
#             print('The shape of pool_prompt_embeds_pred is:', pool_prompt_embeds_pred.shape)
#             for j in range(10):
#                 # image = generator.generate(image_embeds_pred.to(dtype=torch.float16), generator=gen)
#                 image = generator.generate(image_embeds_pred.to(dtype=torch.float16), prompt_embeds_pred, pool_prompt_embeds_pred, generator=gen)
#                 # image = generator.generate(image_embeds_pred.to(dtype=torch.float16), vae_latent_pred, prompt_embeds_pred, pool_prompt_embeds_pred, generator=gen)
#                 # image = generator.generate(image_embeds_pred.to(dtype=torch.float16), prompt_embeds_pred, pool_prompt_embeds_pred, low_level_img, generator=gen)
#                 # The shape of generated images is:(512, 512)
#                 # Construct the save path for each image
#                 path = f'{directory}/{categories[k]}/{j}.png'
#                 # Ensure the directory exists
#                 os.makedirs(os.path.dirname(path), exist_ok=True)
#                 # Save the PIL Image
#                 image.save(path)
#                 print(f'Image saved to {path}')

# if __name__ == '__main__':
#     print(time.asctime(time.localtime(time.time())))
#     main()
#     print(time.asctime(time.localtime(time.time())))