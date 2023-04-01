# GNN Classifier
Implementation of Simple Graph Neural Network Classifier for ModelNet10 Dataset (https://modelnet.cs.princeton.edu/)

## Architecture
Network consists of two modules: Message passing module (GAT Layers), and classifier (Linear layers with ReLU nonlinearity).

## Citations
```bibtex
@article{velickovic2017graph,
    title  = {Graph Attention Networks},
    author = {Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio},
    year   = {2017},
    url = {https://arxiv.org/abs/1710.10903}
}
```