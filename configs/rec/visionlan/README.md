English | [中文](README_CN.md)

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

- [VisionLAN](#visionlan)
  - [1. Introduction](#1-introduction)
    - [1.1 VisionLAN](#11-visionlan)
  - [2. Results](#2-results)
    - [2.1 Accuracy](#21-accuracy)
  - [3. Quick Start](#3-quick-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset preparation](#32-dataset-preparation)
    - [3.3 Update yaml config file](#33-update-yaml-config-file)
    - [3.4 Training](#34-training)
    - [3.5 Evaluation](#35-evaluation)
  - [4. Inference](#4-inference)
  - [5. References](#5-references)



## 1. Introduction

### 1.1 VisionLAN

 Visual Language Modeling Network (VisionLAN) [<a href="#references">1</a>] is a text recognion model that learns the visual and linguistic information simultaneously via **character-wise occluded feature maps** in the training stage. This model does not require an extra language model to extract linguistic information, since the visual and linguistic information can be learned as a union.

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindocr-asset/main/images/visionlan_architecture.PNG" width=450 />
</p>
<p align="center">
  <em> Figure 1. The architecture of visionlan [<a href="#5-references">1</a>] </em>
</p>



As shown above, the training pipeline of VisionLAN consists of three modules:

- The backbone extract visual feature maps from the input image;

- The Masked Language-aware Module (MLM) takes the visual feature maps and a randomly selected character index as inputs, and generates position-aware character mask map to create character-wise occluded feature maps;

- Finally, the Visual Reasonin Module (VRM) takes occluded feature maps as inputs and makes prediction under the complete word-level supervision.

While in the test stage, MLM is not used. Only the backbone and VRM are used for prediction.

## 2. Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

### 2.1 Accuracy

According to our experiments, the evaluation results on public benchmark datasets (IC03, IC13, IC15, IIIT, SVT, SVTP, CUTE) is as follow:

<div align="center">

| **Model** | **Context** | **Backbone**|  **Train Dataset** | **Model Params **|**Avg Accuracy** | **Train Time** | **FPS** | **Recipe** | **Download** |
| :-----: | :-----------: | :--------------: | :----------: | :--------: | :--------: |:----------: |:--------: | :--------: |:----------: |
| visionlan  | GTX3090x8-MS2.0-G | resnet45 | MJ+ST| 42.2M | 90.12%  |  11883s/epoch   | 226.67 | [yaml](https://github.com/mindspore-lab/mindocr/blob/main/configs/rec/visionlan/visionlan_resnet45_LF_1_gpu.yaml) | [ckpt]() |
</div>

<details open markdown>
  <div align="center">
  <summary>Detailed accuracy results for six benchmark datasets</summary>

  | **Model** |  **Context** | **IIIT5k_3000** | **IC13_857** |  **SVT** | **IC15_1811** | **SVTP** | **CUTE80** | **Average** |
  | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |:------: |
  | visionlan |  GTX3090x8-MS2.0-G |  0.9537  |  0.9557  |  0.9289  | 0.8283  |  0.8450  |  0.8958|  0.9012|

  </div>

</details>

**Notes:**

- Context: Training context denoted as `{device}x{pieces}-{MS version}-{MS mode}`. Mindspore mode can be either `G` (graph mode) or `F` (pynative mode). For example, `GTX3090x8-MS2.0-G` denotes training on 8 pieces of NVIDIA GTX3090 GPUs using graph mode based on MindSpore version 2.0.0.
- Train datasets: MJ+ST stands for the combination of two synthetic datasets, SynthText(800k) and MJSynth.
- To reproduce the result on other contexts, please ensure the global batch size is the same.
- The models are trained from scratch without any pre-training. For more dataset details of training and evaluation, please refer to [3.2 Dataset preparation](#32-dataset-preparation) section.
- The models and recipes for training on Ascend are coming soon.


## 3. Quick Start

### 3.1 Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindocr#installation) in MindOCR.

### 3.2 Dataset preparation

* Training sets

The authors of VisionLAN used two synthetic text datasets for training: SynthText(800k) and MJSynth. Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN), or download the two LMDB datasets from [BaiduYun](https://pan.baidu.com/s/1_2dqqxW1vDL9t3B-jlAYRw)(password:z0r5), or download the two LMDB datasets from [Openi Platform](https://openi.pcl.ac.cn/ddeng/ocr_datasets_visionlan/datasets).

After download `SynthText.zip` and `MJSynth.zip`, please unzip and place them under `./datasets/train`.

* Evaluation sets

The authors of VisionLAN used six real text datasets for evaluation: IIIT5K Words (IIIT5K_3000) ICDAR 2013 (IC13_857), Street View Text (SVT), ICDAR 2015 (IC15), Street View Text-Perspective (SVTP), CUTE80 (CUTE). Please follow the instructions of the [original VisionLAN repository](https://github.com/wangyuxin87/VisionLAN), or download them from [BaiduYun](https://pan.baidu.com/s/1sUHgM982YiMf9kmtnhfirg) (password:fjyy), or download them from [Openi Platform](https://openi.pcl.ac.cn/ddeng/ocr_datasets_visionlan/datasets).

After download `evaluation.zip`, please unzip and place them under `./datasets`.

The prepared dataset file struture should be:


``` text
datasets
├── evaluation
│   ├── Sumof6benchmarks
│   ├── CUTE
│   ├── IC13
│   ├── IC15
│   ├── IIIT5K
│   ├── SVT
│   └── SVTP
└── train
    ├── MJSynth
    └── SynText
```

### 3.3 Update yaml config file

If the datasets are placed under `./datasets`, there is no need to change the `train.dataset.dataset_root` in the yaml configuration file `configs/rec/visionlan/visionlan_L*.yaml`.

Otherwise, change the following fields accordingly:

```yaml
...
train:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: train                       <--- Update
...
eval:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- Update
    data_dir: evaluation/Sumof6benchmarks <--- Update
...
```

> Optionally, change `train.loader.num_workers` according to the cores of CPU.


### 3.4 Training

The training stages include Language-free (LF) and Language-aware (LA) process, and in total three steps for training:

```text
LF_1: train backbone and VRM, without training MLM
LF_2: train MLM and finetune backbone and VRM
LA: using the mask generated by MLM to occlude feature maps, train backbone, MLM, and VRM
```

We used distributed training for the three steps. For standalone training, please refer to the [recognition tutorial](../../../docs/en/tutorials/training_recognition_custom_dataset.md#model-training-and-evaluation).

```shell
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1_{platform}.yaml
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2_{platform}.yaml
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA_{platform}.yaml
```

The training result (including checkpoints, per-epoch performance and curves) will be saved in the directory parsed by the arg `ckpt_save_dir` in yaml config file. The default directory is `./tmp_visionlan`.

> `{platform}` in the yaml file names denotes either `gpu` or `ascend`.

### 3.5 Evaluation

After all three steps training, change the `system.distribute` to `False` in `configs/rec/visionlan/visionlan_resnet45_LA_{platform}.yaml`, and then start evaluating the  on six evaluation sets seperately:


```shell
model_name="e8"
yaml_file="configs/rec/visionlan/visionlan_resnet45_LA_{platform}.yaml"
training_step="LA"

python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/IIIT5K eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/IC13 eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/SVT eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/IC15 eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/SVTP eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
python tools/eval.py --config $yaml_file --opt eval.dataset.data_dir=evaluation/CUTE eval.ckpt_load_path="./tmp_visionlan/${training_step}/${model_name}.ckpt"
```

The evaluation results will be printed on the screen. Or, users can save the above script into `run_evaluation.sh`, and then run `bash run_evaluation.sh >> eval_res.log`. The evaluation results will be saved in `eval_res.log`.


## 4. Inference

Coming Soon...


## 5. References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
