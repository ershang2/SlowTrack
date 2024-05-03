# SlowTrack

## Parper

SlowTrack: Increasing the Latency of Camera-Based Perception in Autonomous Driving Using Adversarial Examples

Author: Chen Ma*, Ningfei Wang*, Qi Alfred Chen, Chao Shen (*Co-first authors)

This is the code for the paper [SlowTrack: Increasing the Latency of Camera-Based Perception in Autonomous Driving Using Adversarial Examples](https://ojs.aaai.org/index.php/AAAI/article/view/28200/28396) accepted by AAAI 2024.

The arxiv link to the paper: https://arxiv.org/abs/2312.09520

## Installation
### 1. Installing on the host machine
Step1. Install SlowTrack.
```shell
git clone https://github.com/ershang2/SlowTrack
cd SlowTrack
pip install -r requirements.txt
python setup.py develop
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip install cython_bbox
```
## Data
You can find the model and data through [Google Drive](https://drive.google.com/drive/u/0/folders/16dyUawFm3kUTGIr82xPC4p-8yRl5gX08)

## Adversarial Perturbations Generation
```shell
python tools/latency_attack.py -f exps/example/mot/stra_det_s.py -c path/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fuse --source=path/data/ --local_rank=0 --nms=0.45 --conf=0.25
```
## Adversarial Perturbations Evaluation
```shell
python tools/track.py -f exps/example/mot/yolox_s_mix_det.py -c path/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fuse --source=path/data/ --local_rank=0 --nms=0.45 --conf=0.25
```


## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{ma2024slowtrack,
  title={{SlowTrack: Increasing the Latency of Camera-Based Perception in Autonomous Driving Using Adversarial Examples}},
  author={Ma, Chen and Wang, Ningfei and Chen, Qi Alfred and Shen, Chao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4062--4070},
  year={2024}
}
```
