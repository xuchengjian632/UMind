import torch
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torchsummary import summary
import open_clip
cuda_device_count = torch.cuda.device_count()
print("The number of GPU is: ", cuda_device_count)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_type = 'ViT-H-14'
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)


# vlmodel, preprocess_train = clip.load("ViT-L/14", device=device)

img_directory_training = "/media/siat/disk1/BCI_data/THINGS-EEG/image_set/training_images"
img_directory_test = "/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images"

class img_text_data():
    def __init__(self, train=True):
        self.train = train
        self.n_cls = 1654 if train else 200  

        self.img, self.detail_texts, self.class_names = self.load_data()
        
        self.image_text_encoder(self.img, self.detail_texts, self.class_names)

    def load_data(self):
        images = []
        class_names = []

        if self.train:
            img_directory = img_directory_training  
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)
        
        detail_texts = []
        class_names = []
        if self.train:
            with open('./data/detail_caption_train_BLIP2.txt','r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    detail_texts.append(line)

            with open('./data/class_names_train.txt','r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    for i in range(10):
                        class_names.append(line)
        else:
            with open('./data/detail_caption_test_BLIP2.txt','r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    detail_texts.append(line)

            with open('./data/class_names_test.txt','r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    class_names.append(line)
        
        print('The length of images is:', len(images))
        print('The length of detail texts is:', len(detail_texts))
        print('The length of class names is:', len(class_names))
        return images, detail_texts, class_names

    def image_text_encoder(self, images, detail_texts, class_names):
        batch_size = 20
        image_features_list = []
        text_features_list = []
        class_features_list = []
        
        for i in range(0, len(images), batch_size):
            print('The index is:', i)
            if i + batch_size > len(images):
                idx = len(images)
            else:
                idx = i + batch_size
            batch_images = images[i:idx]
            batch_class_names = class_names[i:idx]
            batch_detail_texts =  detail_texts[i:idx]
            
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)
            with torch.no_grad():
                batch_image_features, _ = vlmodel.encode_image(image_inputs, normalize=True)
                batch_image_features = batch_image_features.cpu()
                # batch_image_features = F.normalize(batch_image_features, dim=-1).detach()
            image_features_list.append(batch_image_features)
            
            # get text features
            class_inputs = torch.cat([clip.tokenize(t, truncate=True) for t in batch_class_names]).to(device)
            with torch.no_grad():
                batch_class_features, _ = vlmodel.encode_text(class_inputs, normalize=True)
                batch_class_features = batch_class_features.cpu()
                # batch_text_features = F.normalize(batch_text_features, dim=-1).detach()
            class_features_list.append(batch_class_features)
            
            text_inputs = torch.cat([clip.tokenize(t, truncate=True) for t in batch_detail_texts]).to(device)
            with torch.no_grad():
                batch_text_features, _ = vlmodel.encode_text(text_inputs, normalize=True)
                batch_text_features = batch_text_features.cpu()
                # batch_text_features = F.normalize(batch_text_features, dim=-1).detach()
            text_features_list.append(batch_text_features)
            
        image_features = torch.cat(image_features_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)
        class_features = torch.cat(class_features_list, dim=0)
        print('The shape of image_features is:', image_features.shape)
        print('The shape of text_features is:', text_features.shape)
        print('The shape of class_features is:', class_features.shape)

        features_filename = os.path.join(f'./data/{model_type}_detail_features_train_BLIP2.pt') if self.train else os.path.join(f'./data/{model_type}_detail_features_test_BLIP2.pt')
        torch.save({
            'img_features': image_features,
            'text_features': text_features,
            'class_features': class_features,
        }, features_filename)
        

if __name__ == "__main__":
    # test_dataset = img_text_data(train=False)
    train_dataset = img_text_data(train=True)

