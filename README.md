# MNE(Multiplex Network Embedding)

This is the source code for IJCAI 2018 paper ["Scalable Multiplex Network Embedding"](http://www.cse.ust.hk/~yqsong/papers/2018-IJCAI-MultiplexNetworkEmbedding.pdf).

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
inproceedings{zhang2018MNE,
title={Scalable Multiplex Network Embedding},
author={Zhang, Hongming and Qiu, Liwei and Yi, Lingling and Song, Yangqiu},
booktitle={IJCAI},
year={2018}
}
```
Note that due to the size limitation of the repository, we only provide few small datasets to help you understand our code and reproduce our experiment. You are welcome to download those largest datasets by yourself or use your own dataset.

# Requirement
```
Python 3
networkx >= 1.11
sklearn >= 0.18.1
gensim >= 1.0.1
```
# Dataset
Here we provide Vickers dataset as an example, you can download all the other datasets from [Twitter Higgs](https://snap.stanford.edu/data/higgs-twitter.html)，[Manlio project](http://deim.urv.cat/~manlio.dedomenico/data.php).
You can also use your own multiplex network dateset, as long as it fits the following template.
```
edge_type head tail weight
    r1     n1   n2    1
    r2     n2   n3    1
    .
    .
    .
```
# Model
Before training, you should first create a folder 'Model' with:

    mkdir Model
The embedding model will be automatically saved into that folder.

# Train
To train the embedding model, simply run:
```
python3 train_model.py data/Vickers-Chan-7thGraders_multiplex.edges
```
You can replace the name of provided dataset with  your own dataset.


# Demo
To repeat the experiment in the paper, simply run:
```
python3 main.py data/Vickers-Chan-7thGraders_multiplex.edges
```


# Acknowledgment
We built the training framework based on the original [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html).
We used the code from [LINE](https://github.com/tangjianpku/LINE), [Node2Vec](https://github.com/aditya-grover/node2vec), and algorithm from [PMNE](https://arxiv.org/pdf/1709.03551.pdf) to complete our experiment.

# Others
If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.
