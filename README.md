# SlowTrack

## Installation
### 1. Installing on the host machine
Step1. Install SlowTrack.
```shell
git clone https://github.com/ershang2/SlowTrack
cd SlowTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

## Adversarial Perturbations Generation
```shell
python3 tools/latency_attack.py -f exps/example/mot/stra_det_s.py -c pretrained/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --source=/data/machen/dataset/MOT17/train/MOT17-05-DPM/img1/ --local_rank=0 --nms=0.45 --conf=0.3
```