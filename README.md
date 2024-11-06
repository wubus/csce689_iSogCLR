# iSogCLR PyTorch Implementation

In this repo, we show how to train a self-supervised model by using Global Contrastive Loss (GCL) on a widely used bimodal image-text dataset [CC3M](https://ai.google.com/research/ConceptualCaptions/download).

## Getting Started

Try in Colab: [https://colab.research.google.com/drive/1FTF-cTcW11Gyrwu8uhTZOXgLsjp49Z9W?usp=sharing](https://colab.research.google.com/drive/1FTF-cTcW11Gyrwu8uhTZOXgLsjp49Z9W?usp=sharing)

### Environment

Setting up a new virtual environment with Conda:
````bash
env_name='csce689_proj'
conda create -n "$env_name" python=3.10
conda activate "$env_name"
pip install -r requirements.txt
````

### Training and Evaluation

1. Download the data: [cc3m_subset_100k.tar.gz](https://drive.google.com/file/d/142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT/view?usp=drive_link), a 100k subset of the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) dataset; [mscoco_val.tar.gz](https://drive.google.com/file/d/142tMsnclHTTPpnTXHSeNgTUlBk4She6o/view?usp=drive_link), a 5k subset of the [COCO](https://cocodataset.org/#home) val2014 dataset; [clip_train.tar.gz](https://drive.google.com/file/d/142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3/view?usp=drive_link), captions of the previous datasets; [imagenet/val.tar](https://drive.google.com/file/d/1NXhfhwFy-nhdABACkodgYqm9pomDKE39/view?usp=sharing), [ImageNet](https://www.image-net.org/challenges/LSVRC/index.php) validation set. The code and data should be structured as follows:
    ```
    .
    +--bimodal_exps (code)
    |
    +--clip_train (captions)
    |  +--cc3m_train_subset.json
    |  +--coco_val.json
    |
    +--datasets (images)
    |  +--cc3m_subset_100k
    |  +--mscoco_val
    |  +--imagnet
    |  |  +-- val
    ```
2. To train a model on cc3m, use `run.slurm` if slurm is supported or run
    ```bash
    export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
    export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

    data_path=./datasets
    ann_path=./clip_train
    train_image_root=cc3m_subset_100k/
    data=cc3m
    train_file=${data}_train_subset.json
    gamma=0.8
    epochs=30
    ita_type=sogclr

    CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir output/${ita_type}_${data}_g${gamma}_e${epochs} \
        --init_model \
        --use_amp \
        --ita_type ${ita_type} \
        --tau_init 0.01 \
        --sogclr_gamma ${gamma} \
        --eta_init 0.03 --sched cosine \
        --no-distributed \
        --epochs ${epochs}
    ```
3. To test the performance of a model on MSCOCO and ImageNet, use `eval.slurm` if slurm is supported or run
    ```bash
    export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
    export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

    data_path=./datasets
    ann_path=./clip_train
    train_image_root=cc3m_subset_100k/
    data=cc3m
    train_file=${data}_train_subset.json
    gamma=0.8
    epochs=30
    ita_type=sogclr

    CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir output/eval_${ita_type}_${data}_g${gamma}_e${epochs} \
        --init_model \
        --use_amp \
        --ita_type ${ita_type} \
        --tau_init 0.01 \
        --sogclr_gamma ${gamma} \
        --eta_init 0.03 --sched cosine \
        --no-distributed \
        --epochs ${epochs} \
        --evaluate --checkpoint ./output/${ita_type}_cc3m_g0.8_e30/checkpoint_30.pth
    ```

## Reference
If you find this tutorial helpful, please cite:
```
@inproceedings{qiu2023not,
  title={Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization},
  author={Zi-Hao Qiu, Quanqi Hu, Zhuoning Yuan, Denny Zhou, Lijun Zhang, and Tianbao Yang},
  booktitle={International Conference on Machine Learning},
  pages={TBD},
  year={2023},
  organization={PMLR}
}
```
