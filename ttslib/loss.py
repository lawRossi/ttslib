"""
@Author: Rossi
Created At: 2022-04-17
"""

import torch
from torch.functional import norm
import torch.nn as nn
from torch.nn.modules import loss


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.SmoothL1Loss(beta=0.01)

    def forward(self, mel_targets, duration_targets, pitch_targets, energy_targets,
           outputs, padding_mask):
        mel_preds = outputs["mel_preds"]
        duration_preds = outputs["duration_preds"]
        pitch_preds = outputs["pitch_preds"]
        energy_preds = outputs["energy_preds"]

        mel_mask = self._get_mel_mask(mel_targets, duration_targets)
        mel_targets = mel_targets.masked_select(mel_mask)
        mel_preds = mel_preds.masked_select(mel_mask)
        mel_loss = self.mae(mel_preds, mel_targets)

        mask = ~padding_mask
        duration_preds = duration_preds.masked_select(mask)
        log_duration_targets = torch.log(duration_targets.float() + 1).masked_select(mask)
        duration_loss = self.mse(duration_preds, log_duration_targets)

        total_loss = mel_loss + duration_loss
        
        losses = {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss
        }

        if pitch_targets is not None:
            pitch_preds = pitch_preds.masked_select(mask)
            pitch_targets = pitch_targets.masked_select(mask)
            pitch_loss = self.mse(pitch_preds, pitch_targets)
            losses ["total_loss"] += pitch_loss
            losses["pitch_loss"] = pitch_loss

        if energy_targets is not None:
            energy_preds = energy_preds.masked_select(mask)
            energy_targets = energy_targets.masked_select(mask)
            energy_loss = self.mse(energy_preds, energy_targets)
            losses["total_loss"] += pitch_loss
            losses["energy_loss"] = energy_loss

        return losses

    def _get_mel_mask(self, mel_targets, duration_targets):
        max_seq_len = mel_targets.shape[1]
        device = mel_targets.device
        mask = torch.arange(max_seq_len).to(device) < duration_targets.sum(dim=1, keepdim=True)
        mask = mask.unsqueeze(-1)
        return mask


class UnetSpeechLoss(FastSpeechLoss):

    def forward(self, mel_targets, duration_targets, padding_mask, outputs):
        mel_preds = outputs["mel_preds"]
        duration_preds = outputs["duration_preds"]

        mel_mask = self._get_mel_mask(mel_targets, duration_targets)
        mel_targets = mel_targets.masked_select(mel_mask)
        mel_preds = mel_preds.masked_select(mel_mask)
        mel_loss = self.mae(mel_preds, mel_targets)

        mask = ~padding_mask
        duration_preds = duration_preds.masked_select(mask)
        normed_duration_targets = self._normalize_durations(duration_targets, padding_mask)
        duration_targets = normed_duration_targets.masked_select(mask)
        duration_loss = self.mse(duration_preds, duration_targets)

        content_loss = self.mse(outputs["encodings"], outputs["mel_encodings"])

        losses = {
            "total_loss": mel_loss + duration_loss + content_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "content_loss": content_loss
        }
        
        return losses

    def _normalize_durations(self, durations, padding_mask):
        length = (~padding_mask).sum(dim=1, keepdim=True)
        mean = durations.sum(dim=1, keepdim=True) / length
        vars = ((durations - mean) ** 2).sum(dim=1, keepdim=True) / length
        std = vars.sqrt()
        normed_durations = (durations - mean) / std
        normed_durations = normed_durations.masked_fill(padding_mask, 0)
        return normed_durations


class AligningLoss(nn.Module):

    def __init__(self, use_kl_step, use_hard_alignment_step, current_step):
        super().__init__()
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.mae = nn.SmoothL1Loss(beta=0.01)
        self.use_kl_step = use_kl_step
        self.use_hard_alignment_step = use_hard_alignment_step
        self.current_step = current_step

    def forward(self, inputs):
        log_probs1 = inputs["log_probs1"]
        p_lens = inputs["phoneme_lens"]
        m_lens = inputs["mel_lens"]
        
        if self.current_step >= self.use_hard_alignment_step:
            ctc_loss = 0
            losses = {"total_loss": 0}
        else:
            ctc_loss = 0
            for i in range(log_probs1.shape[0]):
                targets = torch.arange(1, p_lens[i] + 1).unsqueeze(0)
                cur_logprobs = log_probs1[i][:m_lens[i], :p_lens[i]+1]
                ctc_loss += self.ctc(
                    cur_logprobs.unsqueeze(1), targets, m_lens[i:i+1], p_lens[i:i+1]
                )
            ctc_loss /= log_probs1.shape[0]
            losses = {
                "total_loss": ctc_loss,
                "ctc_loss": ctc_loss
            }

        if self.current_step >= self.use_kl_step:
            log_probs2 = inputs["log_probs2"]
            alignments = inputs["alignments"]
            kl_loss = -(alignments * log_probs2).sum() / (log_probs2.shape[0] * log_probs2.shape[1])
            losses["total_loss"] = ctc_loss + kl_loss
            losses["kl_loss"] = kl_loss

        self.current_step += 1

        return losses

    def _get_mel_mask(self, mel_targets, mel_lens):
        max_seq_len = mel_targets.shape[1]
        device = mel_targets.device
        mask = torch.arange(max_seq_len).to(device) < mel_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        return mask
