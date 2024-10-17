# LiVOS: Lite Video Object Segmentation with Gated Linear Matching
Pytorch implementation for paper [LiVOS: Lite Video Object Segmentation with Gated Linear Matching](https://arxiv.org/), arXiv 2024. <br>

Qin Liu<sup>1</sup>, 
Jianfeng Wang<sup>2</sup>, 
Zhengyuan Yang<sup>2</sup>, 
Linjie Li<sup>2</sup>, 
Kevin Lin<sup>2</sup>, 
Marc Niethammer<sup>1</sup>, 
Lijuan Wang<sup>2</sup> <br>
<sup>1</sup>UNC-Chapel Hill, <sup>2</sup> Microsoft
#### [Paper](https://arxiv.org/) | [Project](https://uncbiag.github.io/LiVOS)

## Installation
The code is tested with ``python=3.10``, ``torch=2.4.0``, ``torchvision=0.19.0``.
```
git clone https://github.com/uncbiag/LiVOS
cd LiVOS
```
Now, create a new conda environment and install required packages accordingly.
```
conda create -n livos python=3.10
conda activate livos
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Quantitative evaluation
- DAVIS 2017 validation: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) or [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2016 validation: [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2017 test-dev: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6812)
- YouTubeVOS 2018 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/7685)
- YouTubeVOS 2019 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6066)
- LVOS val: [LVOS](https://github.com/LingyiHongfd/lvos-evaluation)
- LVOS test: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/8767)
- MOSE val: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/10703#participate-submit_results)
- BURST: [CodaLab](https://github.com/Ali2500/BURST-benchmark)
