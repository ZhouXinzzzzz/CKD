
# CKD: Contrastive Knowledge Distillation from A Sample-wise Perspective

The official implementation for: [CKD: Contrastive Knowledge Distillation from A Sample-wise Perspective](https://arxiv.org/abs/2404.14109).



### Installation

Environments:

- Python 3.8
- PyTorch 1.10.0
- torchvision 0.11.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### Getting started


1. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/ZhouXinzzzzz/CKD/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, our CKD method.
  python3 tools/train.py --cfg configs/cifar100/ckd/res32x4_res8x4.yaml

  # you can also change settings at command line
  python3 tools/train.py --cfg configs/cifar100/ckd/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```

2. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for instance, our CKD method.
  python3 tools/train.py --cfg configs/imagenet/r34_r18/ckd.yaml
  ```

3. Training on Places365

- Download the `places_teachers.tar` at <https://github.com/ZhouXinzzzzz/CKD/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf places_teachers.tar`.

  ```bash
  # for instance, our CKD method.
  python3 tools/train.py --cfg configs/places365/r34_r18/ckd.yaml
  ```

4. Training on MS-COCO

- see [detection.md](detection/README.md)


5. Extension: Visualizations

- Jupyter notebooks: [tsne](tools/visualizations/tsne.ipynb) and [correlation_matrices](tools/visualizations/correlation.ipynb)


### Custom Distillation Method

1. create a python file at `mdistiller/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `mdistiller/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `mdistiller/engines/cfg.py`

4. create a new config file and test it.

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@misc{zhu2025ckdcontrastiveknowledgedistillation,
      title={CKD: Contrastive Knowledge Distillation from A Sample-wise Perspective}, 
      author={Wencheng Zhu and Xin Zhou and Pengfei Zhu and Yu Wang and Qinghua Hu},
      year={2025},
      eprint={2404.14109},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.14109}, 
}
```


# Acknowledgement

- Thanks for DKD and ReviewKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller) and the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).

