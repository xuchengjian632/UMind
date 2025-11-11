<h1 align="center">UMind: A Unified Multitask Network for Zero-Shot M/EEG Visual Decoding</h1>

This repository is the official implementation of UMind.  [📄 Paper](https://arxiv.org/abs/2509.14772) 

## Abstract

- **Unified Multitask Framework**: We introduce a zero-shot M/EEG-based multitask model for retrieval, classification, and reconstruction, surpassing single-task methods through joint optimization and mutual feature reinforcement.
- **Multimodal Alignment Strategy**: Our approach integrates M/EEG, images, and text, using dual-granularity text fusion to enhance neural-visual and semantic representation learning.
- **Dual-Conditional Diffusion Model**: We separately extract neural visual and semantic features and employ them as dual conditions for guiding image generation, ensuring more comprehensive and accurate reconstruction.

![Framework](framework.png)

<div align="center">The framework of UMind.</div>

![Framework](generation_cases.png)

<div align="center">The reconstruction cases based on EEG.</div>

## Datasets
1. [Things-EEG2](https://osf.io/b83fj/overview)
2. [Things-MEG](https://openneuro.org/datasets/ds004212/versions/2.0.0) 



## Multimodal data preparation
### M/EEG pre-processing
- `./EEG-preprocessing/`
- `./MEG-preprocessing/`
### Image and corresponding text preparation

- coarse-grained and fine-grained text generation

```
python detail_text_generation.py
```

- image and text features from pretrained model

```
python img_text_feature_load.py
```

### Data path
- raw coarse-grained text data: `./data/class_names.txt`
- raw fine-grained text data: `./data/detail_caption.txt`
- proprocessed eeg data: `./Data/Things-EEG2/Preprocessed_data_250Hz/`
- proprocessed image and text data: `ViT-H-14_detail_class_features.pt`



## Visual Decoding
### Environment setup

```
pip install -r requirements.txt
```



### Multimodal Alignment Pretraining

```
python EEG_image_retrieval_classification.py
```



### Visual Reconstruction

1. Semantic guidance:

```
python text_condition.py
python text_pool_condition.py
```

2. Visual guidance:

```
python image_condition.py
```

3. EEG-based visual reconstruction

```
python EEG_image_generation.py
```

4. Reconstruction metrics computation

```
python recon_metrics.py
```

## Citation

Hope this code is helpful. I would appreciate you citing us in your paper. 😊
```
@article{xu2025umind,
  title={{UMind}: {A} {Unified} {Multitask} {Network} for {Zero-Shot} {M/EEG} {Visual} {Decoding},
  author={Xu, Chengjian and Song, Yonghao and Liao, Zelin and Zhang, Haochuan and Wang, Qiong and Zheng, Qingqing},
  journal={arXiv preprint arXiv:2509.14772},
  year={2025}
}
```
