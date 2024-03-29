"""
@Author: Rossi
Created At: 2022-04-17
"""

import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.SmoothL1Loss(beta=0.001)

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
            losses["total_loss"] += pitch_loss
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

    def forward(self, outputs):
        mel_targets = outputs["mels"]
        duration_targets = outputs["durations"]
        mel_preds = outputs["mel_preds"]
        duration_preds = outputs["duration_preds"]

        mel_mask = self._get_mel_mask(mel_targets, duration_targets)
        mel_targets = mel_targets.masked_select(mel_mask)
        mel_preds = mel_preds.masked_select(mel_mask)
        mel_loss = self.mae(mel_preds, mel_targets)

        padding_mask = outputs["padding_mask"]
        mask = ~padding_mask
        duration_preds = duration_preds.masked_select(mask)
        normed_duration_targets = self._normalize_durations(duration_targets, padding_mask)
        duration_targets = normed_duration_targets.masked_select(mask)
        duration_loss = self.mse(duration_preds, duration_targets)

        phonme_encodings = outputs["encodings"].masked_select(mel_mask)
        mel_encodings = outputs["mel_encodings"].masked_select(mel_mask)
        content_loss = self.mse(phonme_encodings, mel_encodings)

        losses = {
            "total_loss": mel_loss + duration_loss + content_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "content_loss": content_loss
        }

        mel_preds_ = outputs.get("mel_preds_")
        if mel_preds_ is not None:
            mel_preds_ = mel_preds_.masked_select(mel_mask)
            mel_loss_ = self.mae(mel_preds_, mel_targets)
            losses["total_loss"] += mel_loss_
            losses["mel_loss_"] = mel_loss_

        return losses

    def _normalize_durations(self, durations, padding_mask):
        length = (~padding_mask).sum(dim=1, keepdim=True)
        mean = durations.sum(dim=1, keepdim=True) / length
        vars = ((durations - mean) ** 2).sum(dim=1, keepdim=True) / length
        std = vars.sqrt()
        normed_durations = (durations - mean) / std
        normed_durations = normed_durations.masked_fill(padding_mask, 0)
        return normed_durations
