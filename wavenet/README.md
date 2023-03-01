# WaveNet
Implementation of WaveNet model along with its extention to model MIDI sequences

## Application in MIDI sequence generation
WaveNet is commonly applied to raw audio, but I thought that after some modifications it could be also used in generating MIDI sequences. MelodyWaveNet is wrapper around WaveNet that takes 4 input tensors **pitch**, **velocity** with values [0, 128] and **duration**, **step** with values [0, 7] corresponding to different note lengths (0 - 0 len, 1 - 32th note, 2 - 16th note, 3 - 8th note, etc.). Those tensors are embedded using 4 different embedding layers and then concatenated. After reshaping to (batch_size, 4*embedding_dims, seq_len) shape tensor is passed through WaveNet. WaveNet output is passed through 4 different output blocks each corresponding to different feature (pitch, velocity, duration, step).

```bibtex
@article{vandenoord2016,
    title  = {WaveNet: A Generative Model for Raw Audio},
    author = {Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior and Koray Kavukcuoglu},
    year   = {2016},
    url = {https://arxiv.org/abs/1609.03499}
}
```