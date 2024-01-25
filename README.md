# [AccFlow: Backward Accumulation for Long-Range Optical Flow](https://arxiv.org/abs/2308.13133)

> ICCV2023 | AccFlow: Backward Accumulation for Long-Range Optical Flow  
> Guangyang Wu, Xiaohong Liu, Kunming Luo, Xi Liu, Qingqing Zheng, Shuaicheng Liu, Xinyang Jiang, Guangtao Zhai, Wenyi Wang

## TODO:
- Add inference and visualization codes.
- Add a demo video for better understanding.
- Add figures and brief introduction of this work.
- Provide google drive link for CVO dataset.
- Add *warmstart* mode for evaluation.
- Add evaluation using GMFlow models.

## Requirements
```shell
conda env create -f environment.yml
conda activate accflow
```

## Models
We provide pretrained [models](https://drive.google.com/drive/folders/1-JP8WfNcoaJ1OQMdAPNKMXGfVgf__sso?usp=sharing). The default path of the models for evaluation is:
```Shell
├── checkpoints
    ├── acc+raft-things.pth
    ├── acc+gma-things.pth
    ├── acc+raft-cvo.pth
    ├── acc+gma-cvo.pth
    ├── raft-cvo.pth
    ├── gma-cvo.pth
    ├── raft-things.pth
    ├── gma-things.pth
```

## Evaluation
Download `checkpoints` and put it in the root dir. 

Download testing dataset [CVO-test](data/README.md), put the files `cvo-test.lmdb` and `cvo-test.lmdb-lock` in the directory `data/datasets`.

To evaluate on the clean and final splits, use '-d' param to specify. To evaluate direct methods (e.g., RAFT, GMA), set '-acc' to 'direct'. To evaluate accumulation methods (i.e., accflow), set '-acc' to 'acc'.

```shell
python test_cvo.py -d clean -acc direct -ofe raft --ofe_ckpt checkpoints/raft-things.pth
python test_cvo.py -d clean -acc acc -ofe raft --acc_ckpt checkpoints/acc+raft-things.pth
```

More samples can be found in [test_cvo.sh](test_cvo.sh).


## Training
The script will load the config according to the training stage. The trained model will be saved in a directory in `logs` and `checkpoints`. For example, the following script will load the config `configs/***.yml`.
```shell
# Fine-tune RAFT and GMA (pretrained on flyingthings) using CVO training set
python fine_tune.py -c configs/RAFT.yml
python fine_tune.py -c configs/GMA.yml

# Train AccFlow based on RAFT and GMA (pretrained on flyingthings) using CVO training set
python train_acc.py -c configs/AccGMA.yml
python train_acc.py -c configs/AccRAFT.yml

# Train AccFlow based on RAFT and GMA (fine-tuned with CVO-train)
python train_acc.py -c configs/AccGMA-CVO.yml
python train_acc.py -c configs/AccRAFT-CVO.yml
```

## License
AccFLow is released under the MIT License

## Citation
```bibtex
If you use any part of this code, please kindly cite
@article{wu2023accflow,
  title={AccFlow: Backward Accumulation for Long-Range Optical Flow},
  author={Guangyang Wu and Xiaohong Liu and Kunming Luo and Xi Liu and Qingqing Zheng and Shuaicheng Liu and Xinyang Jiang and Guangtao Zhai and Wenyi Wang},
  journal={arXiv preprint arXiv:2308.13133},
  year={2023}
}
```

## Acknowledgement

In this project, we use parts of codes in:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
