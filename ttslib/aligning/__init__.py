"""
@Author: Rossi
Created At: 2022-06-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttslib.aligning.align import compute_pairwise_distances
from ttslib.layers import GLUBlock


class PhonemeEncoder(nn.Module):

    def __init__(self, embed_dims, kernel_sizes, layers=4, dropout=0.1):
        super().__init__()
        self.encoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_sizes[i], dropout)
                for i in range(layers)]
        )

    def forward(self, phneme_embeddings, padding_mask):
        encodings = phneme_embeddings
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)

        return encodings


class MelEncoder(nn.Module):

    def __init__(self, mel_dims, embed_dims, conv_kernel_size, kernel_sizes, layers=4, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(mel_dims, embed_dims, conv_kernel_size, padding=(conv_kernel_size-1)//2)
        self.encoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_sizes[i], dropout)
                for i in range(layers)]
        )

    def forward(self, mels, padding_mask):
        encodings = self.conv(mels.transpose(2, 1)).transpose(2, 1)
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)

        return encodings


class AligningModel(nn.Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.blank_index = 0
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.phoneme_encoder = PhonemeEncoder(embed_dims, [3, 3, 3, 3])
        self.mel_encoder = MelEncoder(80, embed_dims, 3, [3, 3, 3, 3])

    def forward(self, mels, phonemes, mel_lens, phoneme_lens):
        mel_padding_mask = torch.arange(mels.shape[1]) >= mel_lens.unsqueeze(-1)
        mel_encodings = self.mel_encoder(mels, mel_padding_mask)

        phonemes = F.pad(phonemes, [1, 0, 0, 0], value=self.blank_index)
        phonme_padding_mask = torch.arange(phonemes.shape[1]) > phoneme_lens.unsqueeze(-1)
        phoneme_embeddings = self.embedding(phonemes)
        phoneme_encodings = self.phoneme_encoder(phoneme_embeddings, phonme_padding_mask)

        distances = compute_pairwise_distances(phoneme_encodings, mel_encodings, phoneme_lens, mel_lens)
        attention_scores = torch.softmax(distances, dim=2)

        context = torch.bmm(attention_scores, phoneme_encodings)

        encodings = torch.cat([mel_encodings, context], dim=2)

        print(encodings.shape)


if __name__ == "__main__":
    mels = torch.rand((2, 30, 80))

    phonemes = torch.tensor([
        [2, 4, 4, 5, 4, 6, 1, 1],
        [2, 3, 5, 6, 3, 3, 4, 1]
    ], dtype=torch.long)

    mel_lens = torch.tensor([28, 29])
    phoneme_lens = torch.tensor([6, 7])

    model = AligningModel(30, 20)

    model(mels, phonemes, mel_lens, phoneme_lens)
