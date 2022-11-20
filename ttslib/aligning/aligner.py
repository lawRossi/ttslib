"""
@Author: Rossi
Created At: 2022-07-28
"""

import os

import librosa
import pickle
import torch
import yaml

from ttslib.aligning.model import AligningModel
from ttslib.audio_processing import TacotronSTFT, get_mel_from_audio


class Aligner:
    def __init__(self, model_dir):
        config_file = os.path.join(model_dir, "model_config.yaml")
        model_config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.model = AligningModel(model_config)

        checkpoit = torch.load(os.path.join(model_dir, "checkpoint.pt"))
        self.model.load_state_dict(checkpoit["model"])
        self.model.eval()

        self.stft = TacotronSTFT(
            model_config["mel"]["filter_length"],
            model_config["mel"]["hop_length"],
            model_config["mel"]["win_length"],
            model_config["mel"]["mel_channels"],
            model_config["mel"]["sampling_rate"],
            model_config["mel"]["fmin"],
            model_config["mel"]["fmax"],
        )

        with open(os.path.join(model_dir, "phonemes.pkl"), "rb") as fi:
            self.phoneme_field = pickle.load(fi)

        self.sampling_rate = model_config["mel"]["sampling_rate"]
        self.blank = "<SP>"

    def compute_durations_with_audio(self, audio_file, phonemes):
        audio, _ = librosa.load(audio_file, sr=self.sampling_rate)
        mel, _ = get_mel_from_audio(audio, self.stft)
        mel = mel.transpose(1, 0)
        return self.compute_durations_with_mel(mel, phonemes)

    def compute_durations_with_mel(self, mel, phonemes):
        phonemes_with_blank = [self.blank]
        for phoneme in phonemes.split(" "):
            phonemes_with_blank.append(phoneme)
            phonemes_with_blank.append(self.blank)
        mels = torch.tensor([mel])
        mel_lens = torch.tensor([mels.shape[1]])
        phoneme_tokens = [self.phoneme_field.preprocess(phonemes_with_blank)]
        phonemes = self.phoneme_field.process(phoneme_tokens)
        phoneme_lens = torch.tensor([len(phoneme_tokens[0])])
        with torch.no_grad():
            durations = self.model.inference(mels, phonemes, mel_lens, phoneme_lens)
        phoneme_durations = []
        for phoneme, duration in zip(phonemes_with_blank, durations[0]):
            if duration != 0:
                phoneme_durations.append((phoneme, duration))

        return phoneme_durations


if __name__ == "__main__":
    import json

    aligner = Aligner("data/align_model")

    with open("data/alignments.json") as fi:
        n = 0
        mae_word = 0
        mae_sentence = 0
        for line in fi:
            sample = json.loads(line)
            phonemes, target_durations = zip(*[(p, d) for p, d in zip(sample["phonemes"], sample["durations"]) if p != "sp"])
            target_durations = [sum(target_durations[i:i+2]) for i in range(0, len(target_durations), 2)]
            phonemes = " ".join(phonemes)
            result = aligner.compute_durations_with_audio(sample["audio_file"], phonemes)
            _, durations = zip(*[(p, d) for p, d in result if p != "<SP>"])
            durations = [sum(durations[i:i+2]) for i in range(0, len(durations), 2)]
            assert len(target_durations) == len(durations)
            mae_word += sum(abs(d1 - d2) for d1, d2 in zip(target_durations, durations)) / len(durations)
            mae_sentence += abs(sum(target_durations) - sum(durations))
            n += 1
            if n == 50:
                break
        print(mae_word / n)
        print(mae_sentence / n)

    # mel = np.load("mels/38849211.npy").transpose(1, 0)
    # result = aligner.compute_durations_with_mel(mel, "ni hao")
    # print(result)
