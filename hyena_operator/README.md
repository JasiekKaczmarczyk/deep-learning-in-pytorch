# Hyena Operator
Implementation of Hyena Operator for modelling long sequences without any attention blocks

## Usage
```python
from hyena import HyenaOperator

hyena = HyenaOperator(embedding_size=128, order=4, window_scaling_factor=0.1, window_bias=0.1, causal = True)

# shapes: [batch_size, sequence_length, embedding_size]
x = torch.randn((1, 4096, 128))
y = hyena(x)
```
## Citations
```bibtex
@article{poli2023hyena,
    title  = {Hyena Hierarchy: Towards Larger Convolutional Language Models},
    author = {Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon and Christopher Ré},
    year   = {2023},
    url = {https://arxiv.org/abs/2302.10866}
}
```