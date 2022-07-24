"""
@Author: Rossi
Created At: 2022-07-24
"""

import math

import numpy as np
import torch
import torch.nn as nn

from ttslib.aligning.align import compute_masks


class AligningLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ctc = nn.CTCLoss(zero_infinity=True)

    def forward(self, inputs):
        logits = inputs["logits"]
        log_probs = torch.log_softmax(logits, dim=2).transpose(0, 1)
        phonemes = inputs["phonemes"]
        phoneme_lens = inputs["phoneme_lens"]
        mel_lens = inputs["mel_lens"]

        input_lens = 2 * phoneme_lens + 1
        ctc_loss = self.ctc(log_probs, phonemes, input_lens, phoneme_lens)

        alignments = inputs["alignments"].transpose(1, 2)

        masks = self._compute_alignment_masks(alignments, input_lens, mel_lens)
        alignment_loss = -(alignments * masks).sum() / alignments.shape[0]

        losses = {
            "total_loss": ctc_loss + alignment_loss,
            "ctc_loss": ctc_loss,
            "alignment_loss": alignment_loss
        }

        return losses

    def _compute_alignment_masks(self, alignments, p_lens, m_lens):
        masks = np.zeros(alignments.shape)

        for idx in range(masks.shape[0]):
            mean_len = m_lens[idx] / p_lens[idx]
            width = int(mean_len * 1.5)
            for i in range(p_lens[idx]):
                start = max(0, math.floor(i * mean_len - width))
                end = min(m_lens[idx], math.ceil(i * mean_len + width))
                for j in range(start, end):
                    masks[idx][i][j] = 1

        return torch.tensor(masks, dtype=torch.int)


class OldAligningLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.mse = nn.MSELoss()

    def forward(self, inputs):
        log_probs = inputs["log_probs"]
        phonemes = inputs["phonemes"]
        p_lens = inputs["phoneme_lens"]
        m_lens = inputs["mel_lens"]

        ctc_loss = 0
        for i in range(log_probs.shape[0]):
            cur_logprobs = log_probs[i][:m_lens[i], :p_lens[i]+1]
            ctc_loss += self.ctc(
                cur_logprobs.unsqueeze(1), phonemes[i][:p_lens[i]], m_lens[i:i+1], p_lens[i:i+1]
            )
        ctc_loss /= log_probs.shape[0]
        losses = {
            "total_loss": ctc_loss,
            "ctc_loss": ctc_loss
        }

        mask = (~inputs["mel_padding_mask"]).unsqueeze(-1)
        mel_targets = inputs["mels"].masked_select(mask)
        mel_preds = inputs["mel_preds"].masked_select(mask)
        mel_loss = self.mse(mel_preds, mel_targets)
        losses["total_loss"] = losses["total_loss"] + mel_loss
        losses["mel_loss"] = mel_loss

        return losses
