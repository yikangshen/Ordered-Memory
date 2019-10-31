# Ordered Memory

This repository contains the code used for [Ordered Memory](https://arxiv.org/abs/1910.13466).

The code comes with instructions for experiments:
+ [propositional logic experiments](https://www.aclweb.org/anthology/W15-4002.pdf)

+ [ListOps](https://arxiv.org/pdf/1804.06028.pdf)

+ [SST](https://nlp.stanford.edu/sentiment/treebank.html)

If you use this code or our results in your research, please cite as appropriate:

```
@inproceedings{shen2019ordered,
  title={Ordered Memory},
  author={Shen, Yikang and Tan, Shawn and Hosseini, Seyedarian 
          and Lin, Zhouhan and Sordoni, Alessandro and Courville, Aaron},
  booktitle={Proceedings of the NeurIPS-2019},
  year={2019}
}
```

## Software Requirements

Python 3, PyTorch 1.2, and torchtext are required for the current codebase.

## Experiments

### Propositional Logic

+ `python -u proplog.py --cuda --save logic.pt`

### ListOps

+ `python -u listops.py --cuda --name listops.pt`

### SST

+ `python -u main.py --subtrees --cuda --name sentiment.pt --glove/--elmo (--fine-grained)`

