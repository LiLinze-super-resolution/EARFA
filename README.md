## Efficient Single Image Super-Resolution with Entropy Attention and Receptive Field Augmentation

## Dependencies

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'DAT'.
git clone https://github.com/lilinze-super-rerolution/EARFA.git
conda create -n EARFA python=3.8
conda activate EARFA
pip install -r requirements.txt
python setup.py develop
```

## Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                         |                 Testing Set                 |
|:-----------------------------------------------------|:-------------------------------------------:|
| DIV2 (800 training images, 100 validation images)    | Set5 + Set14 + BSD100 + Urban100 + Manga109 |

- You can download Train datasets and test datasets from [datasets](https://www.aliyundrive.com/s/ZDUy4uLfCVF), but if you want to train EARFA by this code, you need to refer [BasicSR](https://github.com/XPixelGroup/BasicSR) to make sub-datasets form datasets, which you download.
- Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.

## Models

| Method     | Params | FLOPs (G) | Dataset  | PSNR (dB) |  SSIM  |
|:-----------|:------:|:---------:| :------: |:---------:|:------:| 
| EARFA-tiny |  635   |   35.5    | Urban100 |   26.44   | 0.7971 |
| EARFA      |  1045  |   58.4    | Urban100 |   26.68   | 0.8039 |

- You can download pretrained models for testing from [models](https://drive.google.com/drive/folders/13XDUUbskMHwEwCbGbeT8k7g4EamrWLXb?usp=sharing).
- The performance is reported on Urban100 (x4). 'FLOPs' output size of FLOPs is 3×1280×720. 

## Training

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell

  # EARFA, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_EARFA_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_EARFA_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_EARFA_x4.yml --launcher pytorch
  ```
- If you want to train EARFA-tiny, taking X4 as an example, you need to set the num_blocks of 'train_EARFA_x4.yml' to 7.
- The training experiment is in `experiments/`.

## Testing

### Test images with HR

- Download the pre-trained [models](google_dirve) and place them in `experiments/pretrained_models/`.

- We provide pre-trained models for image SR: EARFA and EARFA-tiny.

- Run the following scripts. The testing configuration is in `options/Test/` (e.g., [test_EARFA_x2.yml](options/Test/test_EARFA_x2.yml)).

  ```shell
  
  python basicsr/test.py -opt options/Test/test_EARFA_light_x2.yml
  python basicsr/test.py -opt options/Test/test_EARFA_light_x3.yml
  python basicsr/test.py -opt options/Test/test_earfa_light_x4.yml
  ```

- The output is in `results/`.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
