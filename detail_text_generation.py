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
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torchsummary import summary
cuda_device_count = torch.cuda.device_count()
print("The number of GPU is: ", cuda_device_count)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# vlmodel, preprocess_train = clip.load("ViT-H/14", device=device)
# export HF_ENDPOINT=https://hf-mirror.com
model_type = 'SDXL-text-encoder'

model_id = "llava-hf/llava-1.5-7b-hf"
llava_model = LlavaForConditionalGeneration.from_pretrained(
    "/media/siat/disk1/code/EEG-to-image/pretrain_model/llava-1.5-7b", 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

img_directory_training = "/media/siat/disk1/BCI_data/THINGS-EEG/image_set/training_images"
img_directory_test = "/media/siat/disk1/BCI_data/THINGS-EEG/image_set/test_images"

class img_text_data():
    def __init__(self, train=True):
        self.train = train
        self.n_cls = 1654 if train else 200  

        self.img, self.class_names = self.load_data()
        
        self.image_text_encoder(self.img, self.class_names)

    def load_data(self):
        images = []
        class_names = []

        if self.train:
            img_directory = img_directory_training  
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        for dir in all_folders:
            try:
                idx = dir.index('_')
                description = dir[idx+1:]  
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
            description = description.replace("_", " ")
            new_description = f"This picture is {description}"
            class_names.append(new_description)

        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)
        return images, class_names

    def image_text_encoder(self, images, class_names):
        batch_size = 20
        print('The length of images is:', len(images))
        print('The length of class_names is:', len(class_names))
        detail_texts = []
        for i in range(0, len(images), batch_size):
            print('The index is:', i)
            batch_images = images[i:i + batch_size]
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please give me a description of this picture."},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            for img in batch_images:
                with torch.no_grad():
                    raw_image = Image.open(img).convert("RGB")
                    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
                    output = llava_model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    text_output = processor.decode(output[0], skip_special_tokens=True)
                    text_output = text_output[65:].replace('\n', ' ').replace('\r', ' ')
                    detail_texts.append(text_output)
                    # print('The llava generation output is:', text_output)  
                    
        if self.train:
            with open('./data/detail_caption_train.txt', 'w') as f:
                for i in range(len(detail_texts)):
                    f.write(f"{detail_texts[i]}\n")
        
            with open('./data/class_names_train.txt', 'w') as f:
                for i in range(len(class_names)):
                    f.write(f"{class_names[i]}\n")
        else:
            with open('./data/detail_caption_test.txt', 'w') as f:
                for i in range(len(detail_texts)):
                    f.write(f"{detail_texts[i]}\n")
        
            with open('./data/class_names_test.txt', 'w') as f:
                for i in range(len(class_names)):
                    f.write(f"{class_names[i]}\n")
        

if __name__ == "__main__":
    # test_dataset = img_text_data(train=False)
    train_dataset = img_text_data(train=True)
