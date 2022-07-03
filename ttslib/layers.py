"""
@Author: Rossi
Created At: 2022-04-17
"""

import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, encoding_dims, num_features) -> None:
        super().__init__()
        self.W_gamma = nn.Linear(encoding_dims, num_features)
        self.W_beta = nn.Linear(encoding_dims, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_gamma.weight, 0.0)
        torch.nn.init.constant_(self.W_gamma.bias, 1.0)
        torch.nn.init.constant_(self.W_beta.weight, 0.0)
        torch.nn.init.constant_(self.W_beta.bias, 0.0)

    def forward(self, speaker_encodings, values):
        gamma = self.W_gamma(speaker_encodings)
        beta = self.W_beta(speaker_encodings)
        normed = self.layer_norm(values)
        scaled_normed = normed * gamma.unsqueeze(1) + beta.unsqueeze(1)

        return scaled_normed


class InstanceNorm(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, padding_mask):
        """
        Args:
            inputs (tensor): B x L x E
        """
        length = (~padding_mask).sum(dim=1, keepdim=True).unsqueeze(1)
        mean = inputs.sum(dim=1, keepdim=True) / length
        var = ((inputs - mean) ** 2).sum(dim=1, keepdim=True) / length
        std = (var + self.eps).sqrt()
        normed = (inputs - mean) / std
        normed = normed.masked_fill(padding_mask.unsqueeze(-1), 0)
        return normed, mean.detach(), std.detach()


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.instance_norm = InstanceNorm()

    def forward(self, inputs, mean, std, padding_mask):
        normed, _, _ = self.instance_norm(inputs, padding_mask)
        scaled = std * normed + mean
        scaled = scaled.masked_fill(padding_mask.unsqueeze(-1), 0)
        return scaled


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_dims, max_seq_len=1000):
        super().__init__()
        self.embed_dims = embed_dims

        pe = torch.zeros((max_seq_len, embed_dims))
        for pos in range(max_seq_len):
            for i in range(0, embed_dims, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_dims)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_dims)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embed_dims)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).to(x.device)

        return x


class QuantizationEmbedding(nn.Module):

    def __init__(self, min_value, max_value, n_bins, embed_dims, method="linear"):
        super().__init__()
        assert method in ["log", "linear"]
        if method == "log":
            self.bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(min_value), np.log(max_value), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.bins = nn.Parameter(
                torch.linspace(min_value, max_value, n_bins - 1),
                requires_grad=False,
            )
        self.embedding = nn.Embedding(n_bins, embed_dims)

    def forward(self, x):
        return self.embedding(torch.bucketize(x, self.bins))


class SepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, kernel_size), padding=(0, padding), groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(2)
        conved = self.conv2(self.conv1(x))
        return conved.squeeze(2).transpose(1, 2).contiguous()


class PositionwiseFeedForward(nn.Module):
    """ A two-conv-layer feed forward module """

    def __init__(self, d_in, d_hid, kernel_sizes, dropout=0.1):
        super().__init__()

        self.conv_1 = SepConv(
            d_in,
            d_hid,
            kernel_size=kernel_sizes[0],
            padding=(kernel_sizes[0] - 1) // 2,
        )

        self.conv_2 = SepConv(
            d_hid,
            d_in,
            kernel_size=kernel_sizes[1],
            padding=(kernel_sizes[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.conv_2(torch.relu(self.conv_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(nn.Module):

    def __init__(self, embed_dims, num_heads, hidden_dims, kernel_sizes, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dims)
        self.pos_ff = PositionwiseFeedForward(embed_dims, hidden_dims, kernel_sizes, dropout)

    def forward(self, x, padding_mask):
        residual = x
        attn_output, _ = self.mha(x, x, x, key_padding_mask=padding_mask)
        normed = self.layer_norm(residual + attn_output)
        masked = normed.masked_fill(padding_mask.unsqueeze(-1), 0)
        encoder_output = self.pos_ff(masked).masked_fill(padding_mask.unsqueeze(-1), 0)
        return encoder_output


class ConvBlock(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = SepConv(input_dims, output_dims, kernel_size, (kernel_size-1)//2)
        self.layer_norm = nn.LayerNorm(output_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        residual = x
        conved = torch.relu(self.conv(self.dropout(x)))
        normed = self.layer_norm(residual + conved)
        output = normed.masked_fill(padding_mask.unsqueeze(-1), 0)

        return output


class GLUBlock(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = SepConv(input_dims, output_dims, kernel_size, (kernel_size-1)//2)
        self.conv2 = SepConv(input_dims, output_dims, kernel_size, (kernel_size-1)//2)
        self.layer_norm = nn.LayerNorm(output_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        residual = x
        v = self.conv1(self.dropout(x))
        u = torch.sigmoid(self.conv2(self.dropout(x)))
        normed = self.layer_norm(residual * (1 - u) + v * u)
        output = normed.masked_fill(padding_mask.unsqueeze(-1), 0)
        return output


class ResCnn(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(input_dims, output_dims, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv1d(output_dims, output_dims, kernel_size, padding=(kernel_size-1)//2)
        )

    def forward(self, x, padding_mask):
        residual = x
        output = self.conv_block(x.transpose(2, 1))
        output = output.transpose(2, 1).contiguous()
        output += residual
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0)

        return output


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dims, kernel_sizes, num_heads=None, hidden_dims=None,
                 layers=4, max_seq_len=300, dropout=0.1, padding_idx=0, model_type="FFT"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dims, padding_idx=padding_idx)
        if model_type == "FFT":
            self.pos_embedding = PositionalEmbedding(embed_dims, max_seq_len)
            self.encoding_blocks = nn.ModuleList(
                [FFTBlock(embed_dims, num_heads, hidden_dims, kernel_sizes, dropout) 
                 for _ in range(layers)]
            )
        elif model_type == "CNN":
            self.encoding_blocks = nn.ModuleList(
                [ConvBlock(embed_dims, embed_dims, kernel_sizes[i], dropout) 
                 for i in range(layers)]
            )
        elif model_type == "GLU":
            self.encoding_blocks = nn.ModuleList(
                [GLUBlock(embed_dims, embed_dims, kernel_sizes[i], dropout)
                 for i in range(layers)]
            )
        self.model_type = model_type
        self.padding_idx = padding_idx

    def forward(self, x):
        encodings = self.embedding(x)
        if self.model_type == "FFT":
            encodings = self.pos_embedding(encodings)
        padding_mask = x == self.padding_idx
        for encoding_block in self.encoding_blocks:
            encodings = encoding_block(encodings, padding_mask)

        return encodings, padding_mask


class Decoder(nn.Module):

    def __init__(self, embed_dims, kernel_sizes, output_dims, num_heads=None, hidden_dims=None,
                 layers=4, max_seq_len=1000, dropout=0.1, model_type="FFT"):
        super().__init__()
        if model_type == "FFT":
            self.pos_embedding = PositionalEmbedding(embed_dims, max_seq_len)
            self.decoding_blocks = nn.ModuleList(
                [FFTBlock(embed_dims, num_heads, hidden_dims, kernel_sizes, dropout) for _ in range(layers)]
            )
        elif model_type == "CNN":
            self.decoding_blocks = nn.ModuleList(
                [ConvBlock(embed_dims, embed_dims, kernel_sizes[i], dropout) for i in range(layers)]
            )
        elif model_type == "GLU":
            self.decoding_blocks = nn.ModuleList(
            [GLUBlock(embed_dims, embed_dims, kernel_sizes[i], dropout) for i in range(layers)]
            )
        self.model_type = model_type
        self.linear = nn.Linear(embed_dims, output_dims)

    def forward(self, encodings, padding_mask):
        """_summary_

        Args:
            encodings (torch.Tensor): tensor with shape (B, L, E)
            padding_mask (torch.Tensor): tensor with shape (B, L)

        Returns:
            decoder output: tensor with shape (B, L, out_dims)
        """
        decoder_output = encodings
        if self.model_type == "fft":
            decoder_output = self.pos_embedding(decoder_output)
        for decoding_block in self.decoding_blocks:
            decoder_output = decoding_block(decoder_output, padding_mask)
        decoder_output = self.linear(decoder_output)

        return decoder_output


class AdaptiveDecoderLayer(nn.Module):

    def __init__(self, embed_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = ResCnn(embed_dims, embed_dims, kernel_size, dropout)
        self.norm_layer = AdaptiveLayerNorm(embed_dims, embed_dims)

    def forward(self, inputs, padding_mask, speaker_encodings=None):
        normed = self.norm_layer(speaker_encodings, inputs)
        conved = self.conv(normed, padding_mask)
        output = conved.masked_fill(padding_mask.unsqueeze(-1), 0)

        return output


class AdaptiveDocoder(nn.Module):

    def __init__(self, embed_dims, kernel_sizes, layers, output_dims, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            AdaptiveDecoderLayer(embed_dims, kernel_sizes[i], dropout)
            for i in range(layers)
        ])

        self.linear = nn.Linear(embed_dims, output_dims)

    def forward(self, inputs, padding_mask, speaker_encodings):
        decoder_output = inputs
        for _, layer in enumerate(self.layers):
            decoder_output = layer(inputs, padding_mask, speaker_encodings)

        return self.linear(decoder_output)


class UnetEncoderLayer(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = ResCnn(input_dims, output_dims, kernel_size, dropout)
        self.norm_layer = InstanceNorm()

    def forward(self, inputs, padding_mask):
        normed, mean, std = self.norm_layer(inputs, padding_mask)
        conved = self.conv(normed, padding_mask)

        return conved, mean, std


class UnetEncoder(nn.Module):
    def __init__(self, mel_dims, embed_dims, kernel_sizes, layers, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(mel_dims, embed_dims)
        self.encoder_layers = nn.ModuleList([
            UnetEncoderLayer(embed_dims, embed_dims, kernel_sizes[i], dropout)
            for i in range(layers)
        ])

    def forward(self, inputs, padding_mask):
        encoder_output = self.projection(inputs)
        means = []
        stds = []
        for _, layer in enumerate(self.encoder_layers):
            encoder_output, mean, std = layer(encoder_output, padding_mask)
            means.insert(0, mean)
            stds.insert(0, std)

        return encoder_output, means, stds


class UnetDecoderLayer(nn.Module):
    def __init__(self, emb_dims, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = ResCnn(emb_dims, emb_dims, kernel_size, dropout)
        self.norm_layer = AdaptiveInstanceNorm()

    def forward(self, inputs, mean, std, padding_mask):
        conved = self.conv(inputs, padding_mask)
        normed = self.norm_layer(conved, mean, std, padding_mask)

        return normed


class UnetDocoder(nn.Module):

    def __init__(self, embed_dims, kernel_sizes, layers, output_dims, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            UnetDecoderLayer(embed_dims, kernel_sizes[i], dropout)
            for i in range(layers)
        ])

        self.linear = nn.Linear(embed_dims, output_dims)

    def forward(self, encodings, means, stds, padding_mask):
        decoder_output = encodings
        for i, layer in enumerate(self.layers):
            decoder_output = layer(decoder_output, means[i], stds[i], padding_mask)

        return self.linear(decoder_output)


class StopGradient(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = torch.clone(x).detach()
        output.requires_grad = False
        return output


class VariancePredictor(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, dropout=0.5, stop_gradient=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            SepConv(input_size, filter_size, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),

            SepConv(filter_size, filter_size, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(filter_size, 1)
        if stop_gradient:
            self.sg = StopGradient()
        self.sg = None

    def forward(self, x, mask=None):
        inputs = self.sg(x) if self.sg is not None else x
        out = self.linear(self.conv_block(inputs)).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        return out


class LengthRegulator(nn.Module):

    def forward(self, encoder_output, durations, decoder_max_seq_len):
        """expand encoder output according to durations

        Args:
            encoder_output (torch.Tensor): shape (B, L_enc, E)
            durations (torch.Tensor): shape (B, L_enc)
            max_decoder_seq_len (int): maximum sequence length of decoder input

        Returns:
            expanded encoder output: tensor with shape (B, L_dec, E)
        """
        transition_matrix = self._compute_transition_matrix(durations, decoder_max_seq_len)
        expanded_encoder_output = torch.bmm(encoder_output.transpose(1, 2), transition_matrix)
        return expanded_encoder_output.transpose(1, 2)

    def _compute_transition_matrix(self, durations, max_seq_len):
        """compute the transition matrix to expand the encoder output

        Args:
            durations (torch.Tensor): shape (B, L_enc)
            max_seq_len (int): maximum sequence length of decoder input

        Returns:
            transition matrix : tensor with shape (B , L_enc, L_dec)
        """
        device = durations.device
        range_tensor = torch.arange(max_seq_len).to(device)
        mask1 = range_tensor < torch.cumsum(durations, dim=1).unsqueeze(-1)
        shifted_durations = durations.roll(1)
        shifted_durations[:, 0] = 0
        mask2 = range_tensor < torch.cumsum(shifted_durations, dim=1).unsqueeze(-1)
        transition_matrix = mask1.masked_fill(mask2, False)
        return transition_matrix.float()


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        embed_dims = model_config["embed_dims"]
        filter_size = model_config["variance_predictor"]["filter_size"]
        kernel_size = model_config["variance_predictor"]["kernel_size"]
        dropout = model_config["variance_predictor"]["dropout"]
        stop_gradient = model_config["stop_gradient"]
        self.duration_predictor = VariancePredictor(embed_dims, filter_size, kernel_size, dropout, stop_gradient)
        self.length_regulator = LengthRegulator()

        self.use_energy = model_config["use_energy"]
        self.use_pitch = model_config.get("use_pitch", True)

        if self.use_energy:
            self.energy_predictor = VariancePredictor(embed_dims, filter_size, kernel_size, dropout, stop_gradient)
            min_energy = model_config["variance_predictor"]["min_energy"]
            max_energy = model_config["variance_predictor"]["max_energy"]
            n_bins = model_config["variance_predictor"]["quantization_bins"]
            method = model_config["variance_predictor"]["quantization_method"]
            self.energy_embedding = QuantizationEmbedding(min_energy, max_energy, n_bins, embed_dims, method)

        if self.use_pitch:
            self.pitch_predictor = VariancePredictor(embed_dims, filter_size, kernel_size, dropout, stop_gradient)
            min_pitch = model_config["variance_predictor"]["min_pitch"]
            max_pitch = model_config["variance_predictor"]["max_pitch"]
            n_bins = model_config["variance_predictor"]["quantization_bins"]
            method = model_config["variance_predictor"]["quantization_method"]
            self.pitch_embedding = QuantizationEmbedding(min_pitch, max_pitch, n_bins, embed_dims, method)  

    def forward(self, encoder_output, durations, pitch=None, energy=None, decoder_seq_len=None, padding_mask=None):

        if self.use_pitch:
            pitch_emb = self.pitch_embedding(pitch)
            pitch_emb = pitch_emb.masked_fill(padding_mask.unsqueeze(-1), 0)
            encoder_output = encoder_output + pitch_emb
            pitch_preds = self.pitch_predictor(encoder_output, padding_mask)
        else:
            pitch_preds = None

        if energy is not None:
            energy_emb = self.energy_embedding(energy)
            energy_emb = energy_emb.masked_fill(padding_mask, 0)
            encoder_output = encoder_output + energy_emb
            energy_preds = self.energy_predictor(encoder_output, padding_mask)
        else:
            energy_preds = None

        duration_preds = self.duration_predictor(encoder_output, padding_mask)

        expanded_encoder_output = self.length_regulator(encoder_output, durations, decoder_seq_len)

        return {
            "encodings": expanded_encoder_output,
            "duration_preds": duration_preds,
            "pitch_preds": pitch_preds,
            "energy_preds": energy_preds
        }

    def inference(self, encoder_output, d_control, p_control, e_control, padding_mask, decoder_max_seq_len,
                  reference_durations=None):
        if self.use_pitch:
            pitch_preds = self.pitch_predictor(encoder_output, padding_mask) * p_control
            pitch_emb = self.pitch_embedding(pitch_preds)
            pitch_emb = pitch_emb.masked_fill(padding_mask.unsqueeze(-1), 0)
            encoder_output = encoder_output + pitch_emb
        if self.use_energy:
            energy_preds = self.energy_predictor(encoder_output, padding_mask) * e_control
            energy_emb = self.energy_embedding(energy_preds)
            energy_emb = energy_emb.masked_fill(padding_mask, 0)
            encoder_output = encoder_output + energy_emb

        duration_preds = self.duration_predictor(encoder_output, padding_mask)
        if reference_durations is not None:
            duration_rounded = self.rectify_durations(duration_preds, reference_durations, padding_mask)
        else:
            duration_rounded = self._clamp_duration_preds(duration_preds, padding_mask, d_control)
            duration_rounded = duration_rounded.masked_fill(padding_mask, 0)
        decoder_seq_len = duration_rounded.sum(axis=1).max().item()
        if decoder_max_seq_len is not None:
            decoder_seq_len = min(decoder_seq_len, decoder_max_seq_len)
        expanded_eocoder_output = self.length_regulator(encoder_output, duration_rounded, decoder_seq_len)

        return expanded_eocoder_output, duration_rounded

    def rectify_durations(self, duration_preds, reference_durations, padding_mask):
        length = (reference_durations != 0).sum(dim=1, keepdim=True)
        mean = reference_durations.sum(dim=1, keepdim=True) / length
        vars = ((reference_durations - mean) ** 2).sum(dim=1, keepdim=True) / length
        std = vars.sqrt()
        duration_preds = duration_preds * std + mean
        duration_rounded = torch.ceil(duration_preds).masked_fill(padding_mask, 0)

        return duration_rounded

    def _clamp_duration_preds(self, duration_preds, padding_mask, d_control=1.0):
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(duration_preds) - 1) * d_control),
            min=1,
        )
        duration_rounded = duration_rounded.masked_fill(padding_mask, 0)
        return duration_rounded


class FCSpeakerEncoder(nn.Module):
    def __init__(self, emb_dims, vector_dims, layers=1):
        super().__init__()
        self.layers = layers
        if layers == 1:
            self.projection = nn.Linear(vector_dims, emb_dims)
        else:
            self.projection = nn.Linear(vector_dims, emb_dims//2)
            self.output = nn.Linear(emb_dims//2, emb_dims)

    def forward(self, speaker_vectors):
        if self.layers == 1:
            return self.projection(speaker_vectors)
        else:
            hidden = self.projection(speaker_vectors)
            return self.output(torch.relu(hidden))


class ConvSpeakerEncoder(nn.Module):
    def __init__(self, emb_dims, vector_dims, kernel_sizes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, emb_dims//2, kernel_sizes[0])
        self.conv2 = nn.Conv1d(1, emb_dims//2, kernel_sizes[1])
        self.pooling1 = nn.MaxPool1d(vector_dims-kernel_sizes[0]+1)
        self.pooling2 = nn.MaxPool1d(vector_dims-kernel_sizes[1]+1)

    def forward(self, speaker_vectors):
        speaker_vectors = speaker_vectors.unsqueeze(1)
        part1 = self.pooling1(self.conv1(speaker_vectors)).squeeze(-1)
        part2 = self.pooling2(self.conv2(speaker_vectors)).squeeze(-1)
        return torch.cat([part1, part2], dim=1)


if __name__ == "__main__":
    ins = InstanceNorm()

    input = torch.rand((2, 2, 4))

    print(input)

    mask = torch.tensor([[0, 0], [0, 0]], dtype=torch.bool)
    out1, m, s = ins(input, mask)

    print(out1)

    adain = AdaptiveInstanceNorm()

    out2 = adain(input, m, s, mask)

    print(out2)
