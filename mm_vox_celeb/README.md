# Multimodal VoxCeleb Dataset

This dataset is built on top of the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dataset. We provide facial attribute annotations, segmentation masks, and artistic drawings for each video.
- Video: Videos are cropped with [video-preocessing](https://github.com/AliaksandrSiarohin/video-preprocessing).
- Label: We provide manually annotated facial attributes for each cropped video. The annotation is provided [here](https://drive.google.com/file/d/1Q-ZxGfhNLlIC0X1cW2riBFZ6cz_3tcjy/view?usp=sharing).
- Text: As described [here](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset/issues/3), the textual descriptions are generated using probabilistic context-free grammar (PCFG) based on the given attributes. Code to generate texts is provided in [here](https://github.com/phymhan/MMVID/blob/master/mm_vox_celeb/make_text.py).
- Segmentation: We run [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) to generate segmentation masks for each cropped video. The script is provided [here](https://github.com/phymhan/face-parsing.PyTorch/blob/f6b22fd9488f57210751593a3342e67e7431d5df/generate_mask.py).
- Drawing: We run [Unpaired Portrait Drawing](https://github.com/yiranran/Unpaired-Portrait-Drawing) to generate artistic drawings for each cropped video. The script to generate drawings is provided [here](https://github.com/phymhan/Unpaired-Portrait-Drawing/blob/1e1fabaca51b8f8f86cb299615cae661d4f834f2/generate_drawing.py).

All processed files are available in [this link](https://drive.google.com/drive/folders/18ebgGGTw0610_SRxiu5M3mdJCZqa-O74?usp=sharing). Separate zip files are available: [videos](https://drive.google.com/file/d/1eG4CkNNqEuLz9LCa2XtesNepa9bsa1TP/view?usp=sharing), [masks](https://drive.google.com/file/d/1Y36Or0pEnLQwn9uyORu9394_EcNpa3gl/view?usp=sharing), [drawings](https://drive.google.com/file/d/15UiX1KtyPPSagLjPhnEpm0ynG8PpMT8u/view?usp=sharing), [texts](https://drive.google.com/file/d/19e-9w-0-5FHwIXJ1CmHSKHli3jVMKkLu/view?usp=sharing), [labels](https://drive.google.com/file/d/1Eta6BrTTtV9vv1Hw05n3qo1uvH-3lB4t/view?usp=sharing), [annotations (json)](https://drive.google.com/file/d/1Q-ZxGfhNLlIC0X1cW2riBFZ6cz_3tcjy/view?usp=sharing).

The processed data needs to be organized in the following way:

```
│MMVID/data/mmvoxceleb/
├──video/
│  ├── id11248#yDqlBD8m_b8#00004.txt#000.mp4/
│  │  ├── 0000000.png
│  │  ├── 0000001.png
│  │  ├── ......
│  ├── id11248#yiNkInm9OKQ#00001.txt#000.mp4/
│  ├── ......
├──txt/
│  ├── id11248#yDqlBD8m_b8#00004.txt#000.mp4.txt
│  ├── id11248#yiNkInm9OKQ#00001.txt#000.mp4.txt
│  ├── ......
├──label/
│  ├── id11248#yDqlBD8m_b8#00004.txt#000.mp4.txt
│  ├── id11248#yiNkInm9OKQ#00001.txt#000.mp4.txt
│  ├── ......
├──mask/
│  ├── id11248#yDqlBD8m_b8#00004.txt#000.mp4/
│  │  ├── 0000000.png
│  │  ├── 0000001.png
│  │  ├── ......
│  ├── id11248#yiNkInm9OKQ#00001.txt#000.mp4/
│  ├── ......
├──draw/
│  ├── style1/
│  │  ├── id11248#yDqlBD8m_b8#00004.txt#000.mp4/
|  │  ├── id11248#yiNkInm9OKQ#00001.txt#000.mp4/
```

The first time you run the dataloader, it will create a cache file (`data/mmvoxceleb_local.pkl`). We also provide a pre-generated cache file [here](https://drive.google.com/file/d/15r1cl8KZvuYN_2BvWrU89nEsfKAfN5kg/view?usp=sharing).
