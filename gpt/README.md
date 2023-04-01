# GPT
Implementation of GPT-1 Architecture

## Usage
```python
from transformer import GPT

gpt = GPT(embedding_size = 128, seq_len = 256, out_classes = 10, num_blocks = 6, heads = 8, ffn_expansion = 4, dropout_rate = 0.3)

# shapes: [batch_size, sequence_length, embedding_size]
x = torch.randn((1, 256, 128))
y = gpt(x)
```
## Citations
```bibtex
@article{radford2018improving,
    title  = {Improving Language Understanding by Generative Pre-Training},
    author = {Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever},
    year   = {2018},
    url = {https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf}
}
```
```bibtex
@article{vaswani2017attention,
    title  = {Attention Is All You Need},
    author = {Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin},
    year   = {2017},
    url = {https://arxiv.org/abs/1706.03762}
}
```