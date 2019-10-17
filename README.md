# Fine-Grained-Image-Captioning
The pytorch implementation for "Fine-Grained Image Captioning with Global-Local Discriminative Objective"

## Requirements: ##
- Python 2.7
- PyTorch 0.2
- Torchvision
- coco-caption (download from: https://github.com/tylin/coco-caption, and place in the root directory)
- Pre-trained Resnet101 model (download from: https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM, and should be placed in data/imagenet_weights/)
- Pre-trained VSE++ model (download from:https://drive.google.com/open?id=1D0Bz5LN6-M4FjH4TAaLeOLkP-D7KkXYe, and placed in ./vse/)

## Download MSCOCO dataset ##
- Download the coco images from http://cocodataset.org/#download. Download 2014 Train images and 2014 Val images, and put them into the train2014/ and val2014/ in the ./image.
Download 2014 Test images, and put them into the test2014/

## Download COCO captions and preprocess them ##
- Download Karpathy's split for coco captions from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip .
Extract dataset_coco.json from the zip file and copy it in to ./data/. Then do:
- python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk

## Pre-extract the image features ##
- python scripts/prepro_feats.py --input_json data/dataset_coco.json --images_root image

## Prepare for Reinforcement Learning ##
- Download Cider from: https://github.com/vrama91/cider
And put "ciderD_token.py" and "ciderD_scorer_token4.py" in the "cider/pyciderevalcap/ciderD/", then
- python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train

## Prepare for CSGD training ##
- Download mscoco_knnlabel.h5 from: https://drive.google.com/open?id=1IFkqZuo3Yh2ywwwcdwVMiJKhWT9rWjvz And put it in the "data/"

## Start training ##
### Training using MLE criterion in the initial 20 epochs ###
- python MLE_trainpro.py --id TDA --caption_model TDA --checkpoint_path RL_TDA

### Training by CS-GD ###
- python CSGD_trainpro.py --id TDA --caption_model TDA --checkpoint_path RL_TDA
- We have provided the pre-trained TDA model (download from:https://drive.google.com/open?id=1OVPY1xvCiNQVZpVsMqz4S6-r--q-Bemq , unzip and placed in .RL_TDA/CSGD)

### Eval ###
- python evalpro.py --caption_model TDA --checkpoint_path RL_TDA

### Self-retrieval Experiment ###
- python generate_random_5000.py  --caption_model TDA --checkpoint_path RL_TDA
- python self_retrieval.py --id TDA --caption_model TDA --checkpoint_path RL_TDA



