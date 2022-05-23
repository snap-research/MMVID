## MMVID<br><sub>Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning (CVPR 2022)</sub>

### [Project](https://snap-research.github.io/MMVID/) | [arXiv](https://arxiv.org/abs/2203.02573) | [PDF](https://arxiv.org/pdf/2203.02573.pdf) | [Dataset](#multimodal-voxceleb-dataset)

<div align="center">
  Generated Videos on Multimodal VoxCeleb
</div>

<div class="gif">
<p align="center">
<img src='images/demo.gif' align="center" width=400>
</p>
</div>

This repo contains the code for training and testing, models, and data for MMVID.

> [**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**](https://snap-research.github.io/MMVID/)<br>
> [Ligong Han](https://phymhan.github.io/), [Jian Ren](https://alanspike.github.io/), [Hsin-Ying Lee](http://hsinyinglee.com/), [Francesco Barbieri](https://fvancesco.github.io/), [Kyle Olszewski](https://kyleolsz.github.io/), [Shervin Minaee](https://sites.google.com/site/shervinminaee/home), [Dimitris Metaxas](https://people.cs.rutgers.edu/~dnm/), [Sergey Tulyakov](http://www.stulyakov.com/)<br>
> Snap Inc., Rutgers University<br>
> CVPR 2022


## MMVID Code 

## CLIP model
Download OpenAI's pretrained CLIP model and place it under `./` (or any other directory that is consistent with arg `--openai_clip_model_path`),

```bash
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

## VQGAN

Code for finetuning VQGAN models is provided in [this repo](https://github.com/phymhan/taming-transformers).

## Multimodal VoxCeleb

For testing, please download [pretrained models](#pretrained-models) and change the path for `--dalle_path` in the scripts.

For quantitative evaluation, append `--eval_mode eval` to each testing command. Output log directory can be changed by appending `--name_suffix _fvd` to add suffix (example [here](scripts/mmvoxceleb/text_to_video/evaluation.sh)).

<details>
  <summary>Text-to-Video</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_to_video/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_to_video/test.sh
  #### For Quantitative Evaluation (FVD and PRD):
    bash scripts/mmvoxceleb/text_to_video/evaluation.sh
</details>

<details>
  <summary>Text Augmentation</summary>

  Text augmentation for better training. To enable using a pretrained RoBERTa model, append `--fixed_language_model roberta-large` to the training/testing command. Note that this feature is only *experimental* and is not very robust.

  To enable text dropout, append `--drop_sentence` to the training command. Text dropout is also compatible with using a RoBERTa. We observed that text dropout genrally improves diversity in the generated videos.

  #### Training:
    bash scripts/mmvoxceleb/text_augement/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_augement/test.sh
</details>

<details>
  <summary>Text and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_and_mask/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>

<details>
  <summary>Text and Drawing</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/text_and_drawing/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/text_and_drawing/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>

<details>
  <summary>Drawing and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/drawing_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/drawing_and_mask/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>

<details>
  <summary>Image and Mask</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>

<details>
  <summary>Text and Partial Image</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>

<details>
  <summary>Image and Video</summary>
  
  #### Training:
    bash scripts/mmvoxceleb/image_and_mask/train.sh
  #### Testing:
    bash scripts/mmvoxceleb/image_and_mask/test.sh
  <!-- #### For Quantitative Evaluation (FVD and PRD):
    To Add -->
</details>


## Pretrained Models


Pretrained models are provided [here](https://drive.google.com/drive/folders/1q_YdEBylrAWeuSleq6Jp58epE3KM-oXK?usp=sharing).

### Multimodal VoxCeleb
|     | Weight | FVD |
| --- | :---: | :---: |
| VQGAN (vae) | [ckpt](https://drive.google.com/file/d/1zaud_h46OUJWMKQtkpwaRvHw5I4_wdpg/view?usp=sharing) | - |
| VQGAN (cvae for image conditiong) | [ckpt](https://drive.google.com/file/d/1XO_QKsI6H6c0ombHjnpMTwkW0M7f7nJv/view?usp=sharing) | - |
| Text-to-Video | [pt](https://drive.google.com/file/d/1kBjpLn8Z11w6RqgsNFt1yWUrENb8S1dB/view?usp=sharing) | 59.46 |
| Text-to-Video (ARTV) | [pt](https://drive.google.com/file/d/1enkF3aquQvi7qgGgk-45iQLjgMNs29Cl/view?usp=sharing) | 70.95 |
| Text and Mask | [pt](https://drive.google.com/file/d/1EHLcQ4aZ3ZuUOgPvFcNKFzDdZKGTm5rb/view?usp=sharing) | - |
| Text and Drawing | [pt](https://drive.google.com/file/d/1-kcnX-NY4pX0SEV4It7404yWtG4fCrdr/view?usp=sharing) | - |
| Drawing and Mask | [pt](https://drive.google.com/file/d/13lMHqVVHUfpVqM4edyc3dKeBSFfUKBuq/view?usp=sharing) | - |
| Image and Mask | [pt](https://drive.google.com/file/d/1vcq8la7kpJFqdswfX_KuincRNI6o0h3C/view?usp=sharing) | - |
| Text and Partial Image | [pt](https://drive.google.com/file/d/1wSBm9erN9VP58m3jRQnB_kBCrXW-RGSg/view?usp=sharing) | - |
| Image and Video | [pt](https://drive.google.com/file/d/1LGYA9i5KRA1L-5DlM9Bubbo9PiH2RqfG/view?usp=sharing) | - |
| Text-Augmentation | [pt](https://drive.google.com/file/d/1q-r2PO8qSGunG9w9CjFRbbI9pLO1g_s-/view?usp=sharing) | - |


## Multimodal VoxCeleb Dataset

[**Multimodal VoxCeleb Dataset**](mm_vox_celeb/README.md) has a total of 19,522 videos with 3,437 various interview situations (453 people). Please see details about how to prepare the dataset in `mm_vox_celeb/README.md`. Preprocessed data is also available [here](https://drive.google.com/drive/folders/18ebgGGTw0610_SRxiu5M3mdJCZqa-O74?usp=sharing).


## Acknowledgement
This code is heavily based on [DALLE-PyTorch](https://github.com/lucidrains/DALLE-pytorch) and uses [CLIP](https://github.com/openai/CLIP), [Taming Transformer](https://github.com/CompVis/taming-transformers), [Precision Recall Distribution](https://github.com/msmsajjadi/precision-recall-distributions), [Frechet Video Distance](https://github.com/google-research/google-research/tree/master/frechet_video_distance), [Facenet-PyTorch](https://github.com/timesler/facenet-pytorch), [Face Parsing](https://github.com/zllrunning/face-parsing.PyTorch), and [Unpaired Portrait Drawing](https://github.com/yiranran/Unpaired-Portrait-Drawing).

The authors thank everyone who makes their code and models available.




## Citation

If our code, data, or models help your work, please cite our [paper](https://arxiv.org/abs/2203.02573):
```BibTeX
@inproceedings{han2022show,
title={Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning},
author={Han, Ligong and Ren, Jian and Lee, Hsin-Ying and Barbieri, Francesco and Olszewski, Kyle and Minaee, Shervin and Metaxas, Dimitris and Tulyakov, Sergey},
booktitle={CVPR},
year={2022}
}
```
