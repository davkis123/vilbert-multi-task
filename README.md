# 12-in-1: Multi-Task Vision and Language Representation Learning

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.html):

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

and [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265):

```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
apt-get install build-essential libcap-dev
conda install ipykernel
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```
Replace tools/refer/refer.py with the refer.py file in the root directory.

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data Setup

git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark/
python setup.py build develop
mv maskrcnn_benchmark/ vilbert-multi-task/


cd vilbert-multi-task/data/
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml

## Visiolinguistic Multi Task Training

### Multi-task Training

Download this link which is the trained model for vilbert we are going to use.
```
cd vilbert-multi-task/
```
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin)

Can improve upon our performance by running the following code for fine-tuning on our visual entailment task.
### Fine-tune from Multi-task trained model

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model
```
 
## Run the Feature Extractor 
```
python worker.py
```

## Flask Rest API
```
python app.py
```
while it is running make the post requests:
```

!curl --form "fileupload=@<image_path>" --form text_input=<caption> http://127.0.0.1:5000/
 
```
You can now run and obtain your scores:
```
python demo.py
```
  
## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.
