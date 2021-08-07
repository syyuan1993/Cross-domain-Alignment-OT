# Introduction

This is the official implementation for BMVC 2020 oral paper [Weakly supervised cross-domain alignment with optimal transport](https://www.bmvc2020-conference.com/assets/papers/0566.pdf).

Please consider citing our paper if you refer to this code in your research.
'''
@article{yuan2020weakly,
  title={Weakly supervised cross-domain alignment with optimal transport},
  author={Yuan, Siyang and Bai, Ke and Chen, Liqun and Zhang, Yizhe and Tao, Chenyang and Li, Chunyuan and Wang, Guoyin and Henao, Ricardo and Carin, Lawrence},
  journal={arXiv preprint arXiv:2008.06597},
  year={2020}
}
'''


Code largely borrowed from [code](https://github.com/kuanghuei/SCAN).

## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* [PyTorch 0.4.0](http://pytorch.org/)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```
* nltk stopword package

## Download data

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```

We refer to the path of extracted files for `data.zip` as `$DATA_PATH` and files for `vocab.zip` to `./vocab` directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example, 

```bash
python vocab.py --data_path data --data_name f30k_precomp
python vocab.py --data_path data --data_name coco_precomp
```


## Training new models
Run `train_dot_OT.py`:

```bash
python train_OT.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/coco_scan/log --model_name runs/OT/log --max_violation --bi_gru --margin=0.12 --alpha=1.5 --data_type=full --learning_rate=0.0002 --num_epochs=30 --lr_update=15
python train_OT.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/coco_scan/log --model_name runs/OT/log --max_violation --bi_gru --margin=0.05 --alpha=0.1 --data_type=full
```



### Evaluate trained models
```bash
from vocab import Vocabulary
import evaluation
evaluation.evalrank_dot_OT("$RUN_PATH/coco_dot/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```
To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.
##
