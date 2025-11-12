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

path = "/media/siat/disk1/code/EEG-to-image/pretrain_model/SDXL-turbo"
# pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, variant="fp16")
pipe.to(device)
# path_adapter = "/media/siat/disk1/code/EEG-to-image/pretrain_model/IP-Adapter"
# pipe.load_ip_adapter(
#     path_adapter,
#     subfolder="sdxl_models", 
#     weight_name="ip-adapter_sdxl_vit-h.safetensors", 
#     torch_dtype=torch.float16)
detail_text = []
class_names = []
def text_features_load(train=True):
    if train:
        with open('./data/detail_caption_train.txt','r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                detail_text.append(line)

        with open('./data/class_names_train.txt','r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                for i in range(10):
                    class_names.append(line)
    else:
        with open('./data/detail_caption_test.txt','r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                detail_text.append(line)

        with open('./data/class_names_test.txt','r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                class_names.append(line)
    
    print('The length of detail texts is:', len(detail_text))
    print('The length of class names is:', len(class_names))
    
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    batch_size = 20
    for i in range(0, len(detail_text), batch_size):
        print('Index:', i)
        batch_texts = detail_text[i:i + batch_size]
        batch_class_names = class_names[i:i + batch_size]
        with torch.no_grad():
            (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=batch_texts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None)
            prompt_embeds = prompt_embeds.cpu()
            pooled_prompt_embeds = pooled_prompt_embeds.cpu()
        
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
    print('The shape of prompt_embeds is:', prompt_embeds.shape)
    print('The shape of pooled_prompt_embeds is:', pooled_prompt_embeds.shape)
    features_filename = os.path.join(f'./data/{model_type}_only_text_embeds_train.pt') if train else os.path.join(f'./data/{model_type}_only_text_embeds_test.pt')
    torch.save({
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
    }, features_filename)


if __name__ == "__main__":
    # text_features_load(train=False)
    text_features_load(train=True)
    