"""
@Author: Rossi
Created At: 2022-07-24
"""

import torch
import torch.nn as nn

from ttslib.aligning.align import compute_pairwise_distances, get_hard_alignments
from .loss import AligningLoss
from ttslib.layers import GLUBlock


class MelEncoder(nn.Module):

    def __init__(self, mel_dims, embed_dims, kernel_sizes, dropout=0.2):
        super().__init__()
        self.projection = nn.Linear(mel_dims, embed_dims)
        self.encoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_size, dropout)
             for kernel_size in kernel_sizes]
        )
        self.lstm = nn.LSTM(embed_dims, embed_dims, batch_first=True, bidirectional=True)

    def forward(self, mels, padding_mask):
        encodings = self.projection(mels)
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)
        encodings, _ = self.lstm(encodings)
        return encodings


class PhonemeEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dims, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dims, padding_idx=1)
        self.lstm = nn.LSTM(embed_dims, embed_dims, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, phonemes):
        embed = self.embedding(phonemes)
        encodings, _ = self.lstm(self.dropout(embed))
        return encodings


class AligningModel(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        vocab_size = model_config["vocab_size"]
        mel_dims = model_config["mel_dims"]
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["mel_encoder"]["kernel_sizes"]
        dropout = model_config["dropout"]

        self.phoneme_encoder = PhonemeEncoder(vocab_size, embed_dims)
        self.mel_encoder = MelEncoder(mel_dims, embed_dims, kernel_sizes)
        self.lstm = nn.LSTM(2*embed_dims, embed_dims, batch_first=True)
        self.linear = nn.Linear(embed_dims, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.loss_func = AligningLoss()

    def forward(self, mels, phonemes_with_blank, phonemes, mel_lens, phoneme_lens):
        mel_padding_mask = torch.arange(mels.shape[1]) >= mel_lens.unsqueeze(-1)
        mel_encodings = self.mel_encoder(mels, mel_padding_mask)

        phoneme_encodings = self.phoneme_encoder(phonemes_with_blank)

        distances = compute_pairwise_distances(phoneme_encodings, mel_encodings, phoneme_lens, mel_lens)
        alignments = torch.softmax(distances, dim=2)

        context = torch.bmm(alignments.transpose(1, 2), mel_encodings)
        hidden, _ = self.lstm(self.dropout(context))
        logits = self.linear(hidden)

        outputs = {
            "phonemes": phonemes,
            "logits": logits,
            "mel_lens": mel_lens,
            "phoneme_lens": phoneme_lens,
            "alignments": alignments
        }

        losses = self.loss_func(outputs)

        return losses


if __name__ == "__main__":
    mels = torch.rand((2, 30, 80))

    phonemes = torch.tensor([
        [2, 4, 1],
        [2, 3, 4]
    ], dtype=torch.long)

    phonemes_with_blank = torch.tensor([
        [0, 2, 0, 4, 0, 0, 1],
        [0, 2, 0, 3, 0, 4, 0]
    ], dtype=torch.long)

    mel_lens = torch.tensor([28, 29])
    phoneme_lens = torch.tensor([2, 3])

    model = AligningModel(30, 80, 10, [3, 3])

    losses = model.forward(mels, phonemes_with_blank, phonemes, mel_lens, phoneme_lens)

    print(losses)
