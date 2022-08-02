import torch
import torch.nn as nn
from wavenet import *

class OutputBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, num_classes: int):
        """
        Output Block for Melody WaveNet

        :param int channels: _description_
        :param int out_channels: _description_
        :param int num_classes: _description_
        """
        super().__init__()

        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels*num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        return self.output(x)

class MelodyWaveNet(nn.Module):
    def __init__(self, embedding_dims: int, channels: int, out_channels: int, kernel_size: int, num_stacks: int, num_residual_blocks_in_stack: int):
        """
        Implementation of WaveNet for generating midi sequences

        :param int embedding_dims: embedding dimension for each feature
        :param int channels: intermediate channels
        :param int out_channels: output channels
        :param int kernel_size: size of filter kernels
        :param int num_stacks: num of stacks
        :param int num_residual_blocks_in_stack: num of gated residual conv blocks in each stack
        """
        super().__init__()

        self.out_channels = out_channels

        # embeddings for features, pitch and velocity have values between 0-128
        # duration and step between 0-7 corresponding to different note lengths -> 0 - 0 length, 1 - 32th note, 2 - 16th note, 3 - 8th note, etc.
        # for simplicity dotted notes and triplets are omitted
        self.embedding_pitch = nn.Embedding(num_embeddings=128, embedding_dim=embedding_dims)
        self.embedding_velocity = nn.Embedding(num_embeddings=128, embedding_dim=embedding_dims)
        self.embedding_duration = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dims)
        self.embedding_step = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dims)

        self.wavenet = WaveNet(4*embedding_dims, channels, channels, kernel_size, num_stacks, num_residual_blocks_in_stack)

        # output shape [batch_size, num_classes*out_channels, seq_len]
        self.output_pitch = OutputBlock(channels, out_channels, num_classes=128)
        self.output_velocity = OutputBlock(channels, out_channels, num_classes=128)
        self.output_duration = OutputBlock(channels, out_channels, num_classes=7)
        self.output_step = OutputBlock(channels, out_channels, num_classes=7)

    def forward(self, pitch: torch.Tensor, velocity: torch.Tensor, duration: torch.Tensor, step: torch.Tensor):
        # pitch, velocity, duration and step have shapes of [batch_size, seq_len]

        # embedding each feature, shapes after embedding -> [batch_size, seq_len, embedding_dim]
        pitch_embed = self.embedding_pitch(pitch)
        velocity_embed = self.embedding_velocity(velocity)
        duration_embed = self.embedding_duration(duration)
        step_embed = self.embedding_step(step)

        # concatenating features on embedding dim -> [batch_size, seq_len, 4*embedding_dim]
        # permuting so embedding is counted as channels -> [batch_size, 4*embedding_dim, seq_len]
        features = torch.cat([pitch_embed, velocity_embed, duration_embed, step_embed], dim=-1).permute(0, 2, 1)

        # passing through WaveNet
        x = self.wavenet(features)

        batch_size, _, seq_len = x.shape

        # shapes after output layers -> [batch_size, num_classes*out_channels, seq_len]
        # reshaping so that shapes are [batch_size, num_classes, channels, seq_len]
        pitch = self.output_pitch(x).reshape(batch_size, 128, self.out_channels, seq_len)
        velocity = self.output_velocity(x).reshape(batch_size, 128, self.out_channels, seq_len)
        duration = self.output_duration(x).reshape(batch_size, 7, self.out_channels, seq_len)
        step = self.output_step(x).reshape(batch_size, 7, self.out_channels, seq_len)

        return pitch, velocity, duration, step