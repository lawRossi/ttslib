"""
@Author: Rossi
Created At: 2022-06-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttslib.aligning.align import compute_pairwise_distances, get_hard_alignments
from ttslib.layers import GLUBlock
from ttslib.loss import AligningLoss


class PhonemeEncoder(nn.Module):

    def __init__(self, embed_dims, kernel_sizes, dropout=0.1):
        super().__init__()
        self.encoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_size, dropout)
             for kernel_size in kernel_sizes]
        )

    def forward(self, phneme_embeddings, padding_mask):
        encodings = phneme_embeddings
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)

        return encodings


class MelEncoder(nn.Module):

    def __init__(self, mel_dims, embed_dims, kernel_sizes, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(mel_dims, embed_dims)
        self.encoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_size, dropout)
             for kernel_size in kernel_sizes]
        )

    def forward(self, mels, padding_mask):
        encodings = self.projection(mels)
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)

        return encodings


class PhonemeDecoder(nn.Module):
    
    def __init__(self, embed_dims, output_dims, kernel_sizes, dropout):
        super().__init__()
        
        self.decoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_size, dropout)
             for kernel_size in kernel_sizes]
        )
        self.linear = nn.Linear(embed_dims, output_dims)
        
    def forward(self, encodings, padding_mask):
        decoder_output = encodings
        for decoder_block in self.decoding_blocks:
            decoder_output = decoder_block(decoder_output, padding_mask)
        decoder_output = self.linear(decoder_output)
        
        return decoder_output


class MelDecoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, kernel_sizes, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dims, hidden_dims)
        self.decoding_blocks = nn.ModuleList(
            [GLUBlock(hidden_dims, hidden_dims, kernel_size, dropout)
             for kernel_size in kernel_sizes]
        )
        self.linear = nn.Linear(hidden_dims, output_dims)

    def forward(self, inputs, padding_mask):
        decoder_output = self.projection(inputs)
        print(decoder_output.shape)
        for decoder_block in self.decoding_blocks:
            decoder_output = decoder_block(decoder_output, padding_mask)
        decoder_output = torch.tanh(self.linear(decoder_output))

        return decoder_output


class AligningModel(nn.Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.blank_index = 0
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.phoneme_encoder = PhonemeEncoder(embed_dims, [3, 3, 3, 3])
        self.mel_encoder = MelEncoder(80, embed_dims, [3, 3, 3, 3])
        self.phoneme_decoder = PhonemeDecoder(2*embed_dims, vocab_size, [3, 3], 0.1)
        self.mel_decoder = MelDecoder(vocab_size, 128, 80, [3, 3], 0.1)
        self.loss_func = AligningLoss()

    def forward(self, mels, phonemes, mel_lens, phoneme_lens):
        mel_padding_mask = torch.arange(mels.shape[1]) >= mel_lens.unsqueeze(-1)
        mel_encodings = self.mel_encoder(mels, mel_padding_mask)

        phonemes_ = F.pad(phonemes, [1, 0, 0, 0], value=self.blank_index)
        phonme_padding_mask = torch.arange(phonemes_.shape[1]) > phoneme_lens.unsqueeze(-1)
        phoneme_embeddings = self.embedding(phonemes_)
        phoneme_encodings = self.phoneme_encoder(phoneme_embeddings, phonme_padding_mask)

        distances = compute_pairwise_distances(phoneme_encodings, mel_encodings, phoneme_lens, mel_lens)
        attention_scores = torch.softmax(distances, dim=2)

        context = torch.bmm(attention_scores, phoneme_encodings)

        encodings = torch.cat([mel_encodings, context], dim=2)
        
        logits = self.phoneme_decoder(encodings, mel_padding_mask)
        log_probs = torch.log_softmax(logits, dim=2)
        
        mel_preds = self.mel_decoder(logits, mel_padding_mask)
        print(mel_preds.shape)

        outputs = {
            "phonemes": phonemes,
            "phoneme_lens": phoneme_lens,
            "mel_lens": mel_lens,
            "log_probs": log_probs,
            "mels": mels,
            "mel_preds": mel_preds,
            "mel_padding_mask": mel_padding_mask
        }

        return self.loss_func(outputs)
    
    def inference(self, mels, phonemes, mel_lens, phoneme_lens):
        mel_padding_mask = torch.arange(mels.shape[1]) >= mel_lens.unsqueeze(-1)
        mel_encodings = self.mel_encoder(mels, mel_padding_mask)

        phonemes_ = F.pad(phonemes, [1, 0, 0, 0], value=self.blank_index)
        phonme_padding_mask = torch.arange(phonemes_.shape[1]) > phoneme_lens.unsqueeze(-1)
        phoneme_embeddings = self.embedding(phonemes_)
        phoneme_encodings = self.phoneme_encoder(phoneme_embeddings, phonme_padding_mask)

        distances = compute_pairwise_distances(phoneme_encodings, mel_encodings, phoneme_lens, mel_lens)
        attention_scores = torch.softmax(distances, dim=2)

        context = torch.bmm(attention_scores, phoneme_encodings)

        encodings = torch.cat([mel_encodings, context], dim=2)

        logits = self.phoneme_decoder(encodings, mel_padding_mask)
        log_probs = torch.log_softmax(logits, dim=2)

        alignments = get_hard_alignments(log_probs, phonemes, phoneme_lens, mel_lens, self.blank_index)

        durations = []
        for i, row in enumerate(alignments.numpy().sum(axis=1)):
            durations.append(row[:phoneme_lens[i]])
        return durations


if __name__ == "__main__":
    mels = torch.rand((2, 30, 80))

    phonemes = torch.tensor([
        [2, 4, 4, 5, 4, 6, 1, 1],
        [2, 3, 5, 6, 3, 3, 4, 1]
    ], dtype=torch.long)

    mel_lens = torch.tensor([28, 29])
    phoneme_lens = torch.tensor([6, 7])

    model = AligningModel(30, 20)

    losses = model(mels, phonemes, mel_lens, phoneme_lens)

    print(losses)
