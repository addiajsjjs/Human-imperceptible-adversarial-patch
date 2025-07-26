## Setup

The code in this project was executed using **Python 3.8.11** and ran on a single A6000 GPU. To install the required dependencies, use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Data

We used the following datasets in our experiments:

- [CASIA-NIRVIS](https://pythonhosted.org/bob.db.cbsr_nir_vis_2/)
- [Oulu-CASIA](https://www.v7labs.com/open-datasets/oulu-casia)
- [BUAA-NIRVIS](https://paperswithcode.com/sota/face-verification-on-buaa-visnir?p=wasserstein-cnn-learning-invariant-features)

Directory structure example

```bash
Datasets/
    ├── NIR and VIS images/
    ├── probe.txt
    ├── gallery.txt
```

`probe.txt` and `gallery.txt` examples

```bash
# CASIA
# probe.txt
s1_NIR_00001_001.bmp
...
# gallery.txt
s1_VIS_00001_001.jpg
...

# BUAA
# probe.txt
1/2.bmp
...
# gallery.txt
1/1.bmp
...

# Oulu-CASIA
# probe.txt
NI_Strong_P001_Disgust_001.jpeg
...
# gallery.txt
VL_Strong_P001_Surprise_001.jpeg
...
```

## Models

All images (both NIR and VIS) should be cropped and resized to **224x224** using **[RetinaFace](https://github.com/serengil/retinaface)**.

Face landmark detection to generate a valid mask is performed using **[Shape Predictor 81](https://github.com/codeniko/shape_predictor_81_face_landmarks)**.

Download the tool models ([ResNeSt, LightCNN, LightCNN-DVG, LightCNN-Rob](https://drive.google.com/file/d/1RWisrZlrrNVRHiElxcr7cH6SU1eyDeiK/view?usp=sharing)) and place them in ``./models/``.

## Example Commands

To run an untargeted attack using LightCNN on the CASIA dataset, execute:

```
CUDA_VISIBLE_DEVICES=0 python start.py \
    --model LightCNN \
    --data_type casia \
    --dataset_path your_dataset_path \
    --probe_file_path your_dataset_probe_file_path \
    --gallery_file_path your_dataset_gallery_file_path \
    --attack_type untargeted_attack \
    --batch_size 512 \
    --workers 64 \
    --AnchorNum 4 \
    --AreaNum 8 \
    --maxiter 200
```
