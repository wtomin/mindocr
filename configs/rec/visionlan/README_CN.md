[English](README.md) | 中文

# VisionLAN

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> VisionLAN: [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)

- [VisionLAN](#visionlan)
  - [1. 简介](#1-简介)
    - [1.1 VisionLAN](#11-visionlan)
  - [2.精度结果](#2精度结果)
  - [3.快速入门](#3快速入门)
    - [3.1安装](#31安装)
    - [3.2数据集准备](#32数据集准备)
    - [3.3 更新yaml配置文件](#33-更新yaml配置文件)
    - [3.4 训练](#34-训练)
    - [3.5 评估](#35-评估)
  - [4. 推理](#4-推理)
  - [5. 引用文献](#5-引用文献)


## 1. 简介

### 1.1 VisionLAN

视觉语言建模网络（VisionLAN）[<a href="#5-引用文献">1</a>]是一种文本识别模型，它通过在训练阶段使用逐字符遮挡的特征图来同时学习视觉和语言信息。这种模型不需要额外的语言模型来提取语言信息，因为视觉和语言信息可以作为一个整体来学习。
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindocr-asset/main/images/visionlan_architecture.PNG" width=450 />
</p>
<p align="center">
  <em> 图 1. Visionlan 的模型结构 [<a href="#5-引用文献">1</a>] </em>
</p>

如上图所示，VisionLAN的训练流程由三个模块组成：

- 骨干网络从输入图像中提取视觉特征图；
- 掩码语言感知模块（MLM）以视觉特征图和一个随机选择的字符索引作为输入，并生成位置感知的字符掩码图，以创建逐字符遮挡的特征图；
- 最后，视觉推理模块（VRM）以遮挡的特征图作为输入，并在完整的单词级别的监督下进行预测。

但在测试阶段，MLM不被使用。只有骨干网络和VRM被用于预测。

## 2.精度结果


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

**注**

- 训练环境表示为`{device}x{pieces}-{MS版本}-{MS模式}`。MindSpore模式可以是`G`（图模式）或`F`（原生模式）。例如，`GTX3090x8-MS2.0-G`表示使用MindSpore版本2.0.0在8块NVIDIA GTX3090 GPU上使用图模式进行训练。
- 训练数据集：`MJ+ST`代表两个合成数据集SynthText（800k）和MJSynth的组合。
- 要在其他训练环境中重现结果，请确保全局批量大小相同。
- 这些模型是从头开始训练的，没有任何预训练。有关训练和评估的更多数据集详细信息，请参阅3.2数据集准备部分。
- 用于Ascend上训练的模型和yaml文件即将推出。


## 3.快速入门

### 3.1安装

请参考[MindOCR中的安装说明]((https://github.com/mindspore-lab/mindocr#installation))。

### 3.2数据集准备

* 训练集
VisionLAN的作者使用了两个合成文本数据集进行训练：SynthText（800k）和MJSynth。请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，或从[BaiduYun](https://pan.baidu.com/s/1_2dqqxW1vDL9t3B-jlAYRw)（密码：z0r5）下载这两个LMDB数据集，或从[Openi Platform](https://openi.pcl.ac.cn/ddeng/ocr_datasets_visionlan/datasets)下载这两个LMDB数据集。

下载`SynthText.zip`和`MJSynth.zip`后，请解压缩并将它们放置在`./datasets/train`目录下。



* 评估集

VisionLAN的作者使用了六个真实文本数据集进行评估：IIIT5K Words（IIIT5K_3000）、ICDAR 2013（IC13_857）、Street View Text（SVT）、ICDAR 2015（IC15_1811）、Street View Text-Perspective（SVTP）、CUTE80（CUTE）。

请按照[原始VisionLAN repository](https://github.com/wangyuxin87/VisionLAN)的说明进行操作，或从[BaiduYun](https://pan.baidu.com/s/1sUHgM982YiMf9kmtnhfirg)（密码：fjyy）下载它们，或从[Openi Platform](https://openi.pcl.ac.cn/ddeng/ocr_datasets_visionlan/datasets)下载它们。


下载`evaluation.zip`后，请解压缩并将其放置在`./datasets`目录下。准备好的数据集文件结构应如下所示：

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

### 3.3 更新yaml配置文件

如果数据集放置在`./datasets`目录下，则无需更改yaml配置文件`configs/rec/visionlan/visionlan_L*.yaml`中的`train.dataset.dataset_root`。
否则，请相应地更改以下字段：


```yaml
...
train:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: train                       <--- 更新
...
eval:
  dataset_sink_mode: False
  dataset:
    type: LMDBDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: evaluation/Sumof6benchmarks <--- 更新
...
```

> 您也可以选择根据CPU的线程数量来修改 `train.loader.num_workers`.


### 3.4 训练

训练阶段包括无语言（LF）和有语言（LA）过程，总共有三个训练步骤：

```text
LF_1：训练骨干网络和VRM，不训练MLM
LF_2：训练MLM并微调骨干网络和VRM
LA：使用MLM生成的掩码遮挡特征图，训练骨干网络、MLM和VRM
```

我们接下来使用分布式训练进行这三个步骤。对于单卡训练，请参考[识别教程](../../../docs/cn/tutorials/training_recognition_custom_dataset.md#单卡训练)。

```shell
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_1_{platform}.yaml
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LF_2_{platform}.yaml
mpirun --allow-run-as-root -n 8 python tools/train.py --config configs/rec/visionlan/visionlan_resnet45_LA_{platform}.yaml
```

训练结果（包括checkpoints、每个阶段的性能和loss曲线）将保存在yaml配置文件中由参数`ckpt_save_dir`解析的目录中。默认目录为`./tmp_visionlan`。

> yaml 配置文件中的`{platform}`指的是 `gpu` 或者 `ascend`.

### 3.5 评估

在完成上述三个训练步骤后，在`configs/rec/visionlan/visionlan_resnet45_LA_{platform}.yaml`中将`system.distribute`更改为`False`，然后分别开始对六个测试集进行评估：


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


评估结果将会被打印在屏幕上。或者，用户可以将上述脚本保存为`run_evaluation.sh`，然后运行`bash run_evaluation.sh >> eval_res.log`。评估结果将保存在`eval_res.log`文件中。

## 4. 推理

敬请期待...


## 5. 引用文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang: From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network. ICCV 2021: 14174-14183
