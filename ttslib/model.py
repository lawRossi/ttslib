"""
@Author: Rossi
Created At: 2022-04-17
"""

import torch
import torch.nn as nn

from ttslib.layers import AdaptiveDocoder, Decoder, Encoder
from ttslib.layers import VarianceAdaptor, UnetDocoder, UnetEncoder
from ttslib.loss import FastSpeechLoss, UnetSpeechLoss


class SynthesisModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self._init_encoder(model_config)
        self._init_decoder(model_config)
        self._init_variance_adaptor(model_config)
        self._init_speaker_embedding(model_config)
        self._init_loss(model_config)
        self.fine_tuning = False

    def _init_encoder(self, model_config):
        pass

    def _init_decoder(self, model_config):
        pass

    def _init_variance_adaptor(self, model_config):
        self.variance_adaptor = VarianceAdaptor(model_config)

    def _init_speaker_embedding(self, model_config):
        embed_dims = model_config["embed_dims"]
        num_speakers = model_config["num_speakers"]
        self.speaker_embedding = nn.Embedding(num_speakers, embed_dims)

    def _init_loss(self, model_config):
        self.loss_func = FastSpeechLoss()

    def start_fine_tuning(self):
        self.fine_tuning = True

    def forward(self, mels, phonemes, speakers, durations, pitch=None, energy=None):
        encoder_output, padding_mask = self.encoder(phonemes)

        speaker_embs = self.speaker_embedding(speakers)
        encodings = encoder_output + speaker_embs.unsqueeze(1)

        decoder_seq_len = mels.shape[1]
        outputs = self.variance_adaptor(
            encodings, durations, pitch, energy, decoder_seq_len, padding_mask
        )
        decoder_padding_mask = self._compute_mel_padding_mask(mels, durations)
        mel_preds = self.decoder(outputs["encodings"], decoder_padding_mask)
        outputs["mel_preds"] = mel_preds

        return self.loss_func(mels, durations, pitch, energy, outputs, padding_mask)

    def inference(self, phonemes, speakers, d_control=1.0, p_control=1.0, e_control=1.0, **kwargs):
        encoder_output, padding_mask = self.encoder(phonemes)

        speaker_embs = self.speaker_embedding(speakers)
        encoder_output = encoder_output + speaker_embs.unsqueeze(1)

        encodings, durations = self.variance_adaptor.inference(
            encoder_output, d_control, p_control, e_control, padding_mask, self.decoder_max_seq_len
        )

        decoder_padding_mask = self._compute_decoder_padding_mask(durations)
        mel_outputs = self.decoder(encodings, decoder_padding_mask)
        mels = []
        for i in range(mel_outputs.shape[0]):
            mel_len = int(durations[i].sum().item())
            mel = mel_outputs[i][:mel_len]
            mels.append(mel)

        return mels

    def _compute_mel_padding_mask(self, mels, durations):
        decoder_seq_len = mels.shape[1]
        device = mels.device
        mel_padding_mask = torch.arange(decoder_seq_len).to(device) >= durations.sum(dim=1, keepdim=True)
        return mel_padding_mask

    def _compute_decoder_padding_mask(self, durations):
        decoder_seq_len = durations.sum(dim=1).max().item()
        if self.decoder_max_seq_len is not None:
            decoder_seq_len = min(decoder_seq_len, self.decoder_max_seq_len)
        device = durations.device
        decoder_padding_mask = torch.arange(decoder_seq_len).to(device) >= durations.sum(dim=1, keepdim=True)

        return decoder_padding_mask


class FastSpeech(SynthesisModel):

    def _init_encoder(self, model_config):
        vocab_size = model_config["vocab_size"]
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["encoder"]["kernel_sizes"]
        num_heads = model_config["encoder"].get("num_heads")
        hidden_dims = model_config["encoder"].get("hidden_dims")
        max_seq_len = model_config["encoder"].get("max_seq_len")
        dropout = model_config["encoder"]["dropout"]
        padding_idx = model_config["encoder"].get("padding_idx", 0)
        model_type = model_config["encoder"]["model_type"]
        self.encoder = Encoder(
            vocab_size, embed_dims, kernel_sizes, num_heads, hidden_dims,
            max_seq_len, dropout, padding_idx, model_type
        )

    def _init_decoder(self, model_config):
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["decoder"]["kernel_sizes"]
        output_dims = model_config["decoder"]["output_dims"]
        num_heads = model_config["decoder"].get("num_heads")
        hidden_dims = model_config["decoder"].get("hidden_dims")
        max_seq_len = model_config["decoder"].get("max_seq_len")
        self.decoder_max_seq_len = max_seq_len
        dropout = model_config["decoder"]["dropout"]
        model_type = model_config["decoder"]["model_type"]
        self.decoder = Decoder(
            embed_dims, kernel_sizes, output_dims, num_heads, hidden_dims, 
            max_seq_len, dropout, model_type
        )


class AdaSpeech(FastSpeech):
    def _init_decoder(self, model_config):
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["decoder"]["kernel_sizes"]
        output_dims = model_config["decoder"]["output_dims"]
        max_seq_len = model_config["decoder"].get("max_seq_len")
        self.decoder_max_seq_len = max_seq_len
        dropout = model_config["decoder"]["dropout"]
        self.decoder = AdaptiveDocoder(
            embed_dims, kernel_sizes, output_dims, dropout
        )

    def forward(self, mels, phonemes, speakers, durations, pitch=None):
        encodings, padding_mask = self.encoder(phonemes)
        speaker_embs = self.speaker_embedding(speakers)
        decoder_seq_len = mels.shape[1]
        outputs = self.variance_adaptor(
            encodings, durations, pitch, None, decoder_seq_len, padding_mask
        )
        decoder_padding_mask = self._compute_mel_padding_mask(mels, durations)
        mel_preds = self.decoder(outputs["encodings"], decoder_padding_mask, speaker_embs)
        outputs["mel_preds"] = mel_preds

        return self.loss_func(mels, durations, pitch, None, outputs, padding_mask)

    def inference(self, phonemes, speakers, d_control=1.0, p_control=1.0, e_control=1.0, **kwargs):
        encoder_output, padding_mask = self.encoder(phonemes)

        speaker_embs = self.speaker_embedding(speakers)

        encodings, durations = self.variance_adaptor.inference(
            encoder_output, d_control, p_control, e_control, padding_mask, self.decoder_max_seq_len
        )

        decoder_padding_mask = self._compute_decoder_padding_mask(durations)
        mel_outputs = self.decoder(encodings, decoder_padding_mask, speaker_embs)
        mels = []
        for i in range(mel_outputs.shape[0]):
            mel_len = int(durations[i].sum().item())
            mel = mel_outputs[i][:mel_len]
            mels.append(mel)

        return mels


class UnetSpeech(FastSpeech):

    def __init__(self, model_config):
        super().__init__(model_config)
        self._init_mel_encoder(model_config)
        self._load_pretrained_encoder(model_config)
        self.use_adv = model_config.get("use_adv", False)

    def _init_decoder(self, model_config):
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["decoder"]["kernel_sizes"]
        output_dims = model_config["decoder"]["output_dims"]
        max_seq_len = model_config["decoder"].get("max_seq_len")
        self.decoder_max_seq_len = max_seq_len
        dropout = model_config["decoder"]["dropout"]
        self.decoder = UnetDocoder(embed_dims, kernel_sizes, output_dims, dropout)

    def _init_mel_encoder(self, model_config):
        embed_dims = model_config["embed_dims"]
        kernel_sizes = model_config["mel_encoder"]["kernel_sizes"]
        mel_dims = model_config["mel_encoder"]["mel_dims"]
        dropout = model_config["mel_encoder"]["dropout"]
        self.mel_encoder = UnetEncoder(mel_dims, embed_dims, kernel_sizes, dropout)

    def _init_loss(self, model_config):
        self.loss_func = UnetSpeechLoss()

    def _load_pretrained_encoder(self, model_config):
        if not model_config["encoder"]["pretrained_model"]:
            return
        pretrained_state_dict = torch.load(model_config["encoder"]["pretrained_model"])["model"]
        state_dict = self.state_dict()
        for key in pretrained_state_dict.keys():
            if key.startswith("encoder"):
                state_dict[key] = pretrained_state_dict[key]
        self.load_state_dict(state_dict)

    def forward(self, mels, phonemes, durations, **kwargs):
        encodings, padding_mask = self.encoder(phonemes)
        decoder_seq_len = mels.shape[1]
        outputs = self.variance_adaptor(
            encodings, durations, None, None, decoder_seq_len, padding_mask
        )
        mel_padding_mask = self._compute_mel_padding_mask(mels, durations)
        mel_encodings, means, stds = self.mel_encoder(mels, mel_padding_mask)

        mel_preds = self.decoder(outputs["encodings"], means, stds, mel_padding_mask)
        if self.use_adv:
            mel_preds_ = self.decoder(mel_encodings.detach(), means, stds, mel_padding_mask)
            outputs["mel_preds_"] = mel_preds_
        outputs["mel_encodings"] = mel_encodings
        outputs["mel_preds"] = mel_preds
        outputs["means"] = means
        outputs["stds"] = stds
        outputs["mels"] = mels
        outputs["durations"] = durations
        outputs["mel_padding_mask"] = mel_padding_mask
        outputs["padding_mask"] = padding_mask

        return self.loss_func(outputs)

    def inference(self, mels, phonemes, durations, d_control=1.0, **kwargs):
        encodings, padding_mask = self.encoder(phonemes)
        encodings, duration_preds = self.variance_adaptor.inference(
            encodings, d_control, None, None, padding_mask, self.decoder_max_seq_len, durations
        )

        mel_padding_mask = self._compute_mel_padding_mask(mels, durations)
        _, means, stds = self.mel_encoder(mels, mel_padding_mask)

        decoder_padding_mask = self._compute_decoder_padding_mask(duration_preds)
        mel_preds = self.decoder(encodings, means, stds, decoder_padding_mask)

        mels = []
        for i in range(mel_preds.shape[0]):
            mel_len = int(duration_preds[i].sum().item())
            mel = mel_preds[i][:mel_len]
            mels.append(mel)

        return mels

    def teacher_forcing_inference(self, mels, durations, phonemes=None):
        mel_padding_mask = self._compute_mel_padding_mask(mels, durations)
        mel_encodings, means, stds = self.mel_encoder(mels, mel_padding_mask)

        if phonemes is not None:
            encodings, padding_mask = self.encoder(phonemes)
            decoder_seq_len = mels.shape[1]
            outputs = self.variance_adaptor(
                encodings, durations, None, None, decoder_seq_len, padding_mask
            )
            mel_preds = self.decoder(outputs["encodings"], means, stds, mel_padding_mask)
        else:
            mel_preds = self.decoder(mel_encodings, means, stds, mel_padding_mask)
        return mel_preds


if __name__ == "__main__":
    import yaml

    # model_config = yaml.load(open("data/ada_model_config.yaml"), Loader=yaml.FullLoader)
    # model = AdaSpeech(model_config)

    model_config = yaml.load(open("data/unet_model_config.yaml"), Loader=yaml.FullLoader)
    model = UnetSpeech(model_config)

    phonemes = torch.tensor([
        [1, 3, 5, 9, 0],
        [1, 2, 4, 0, 0]
    ], dtype=torch.long)

    mels = torch.rand((2, 50, 80))
    speakers = torch.tensor([1, 2], dtype=torch.long)
    durations = torch.tensor([[1, 3, 5, 3, 4], [3, 2, 4, 5, 2]])
    # pitch = torch.rand((2, 5))

    print(model.forward(mels, phonemes, durations))

    print(model.inference(mels, phonemes, durations))
