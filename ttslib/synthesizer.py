"""
@Author: Rossi
Created At: 2022-04-25
"""

import json
import os
import re
import torch
import yaml


import aukit
import numpy as np
import pickle
import pydub

from ttslib import hifigan
from ttslib.text import text2phonemes
from ttslib.utils import find_class


class Synthesizer:
    def __init__(self, model_dir) -> None:
        config_file = os.path.join(model_dir, "model_config.yaml")
        model_config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        model_cls = find_class(model_config["model"])
        self.synthesize_model = model_cls(model_config)
        self.synthesize_model.eval()
        checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        self.synthesize_model.load_state_dict(checkpoint["model"])
        with open(os.path.join(model_dir, "phonemes.pkl"), "rb") as fi:
            self.phoneme_field = pickle.load(fi)

        if model_config["use_existing_speaker_vectors"]:
            self._load_speaker_vectors(model_config)
        else:
            self.speaker_vectors = None
            with open(os.path.join(model_dir, "speakers.pkl"), "rb") as fi:
                self.speaker_field = pickle.load(fi)

        self._init_vocoder(model_config)

        self.sentence_delimiter = re.compile(r"[,。，；;？！?]")
        self.sampling_rate = 22050


    def _load_speaker_vectors(self, model_config):
        with open(model_config["speaker_encoder"]["vector_file"], "rb") as fi:
            self.speaker_vectors = pickle.load(fi)
            for speaker, vectors in self.speaker_vectors.items():
                self.speaker_vectors[speaker] = np.array(vectors).mean(axis=0)
            # self.speaker_vectors = defaultdict(list)
            # for line in fi:
            #     data = json.loads(line)
            #     for speaker, values in data.items():
            #         for _, vector in values:
            #             self.speaker_vectors[speaker].append(vector)

    def _init_vocoder(self, model_config):
        self.vocoder_model = model_config["vocoder"]["model"]
        if self.vocoder_model == "melgan":
            self.vocoder = torch.hub.load(
                "models/melgan", "load_melgan", "multi_speaker", source="local"
            )
            self.vocoder.mel2wav.eval()
        elif self.vocoder_model == "hifigan":
            with open("models/hifigan/config.json") as fi:
                config = json.load(fi)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            # ckpt = torch.load("models/hifigan/generator_universal.pth.tar", map_location="cpu")
            ckpt = torch.load("models/hifigan/generator_v3", map_location="cpu")
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            self.vocoder = vocoder

    def _split_sentences(self, text):
        return [split.strip() for split in self.sentence_delimiter.split(text) if split.strip()]

    def synthesize(self, text, speaker, save_file):
        sentences = self._split_sentences(text)
        print(sentences)
        phonemes = [text2phonemes(sentence) for sentence in sentences]
        phonemes = self.phoneme_field.process(phonemes)

        if self.speaker_vectors is None:
            speakers = self.speaker_field.process([speaker] * len(sentences))
        else:
            speakers = torch.tensor(self.speaker_vectors[speaker], dtype=torch.float).repeat((len(sentences), 1))
        with torch.no_grad():
            mels_list = self.synthesize_model.inference(phonemes, speakers)
        wavs = []
        for mels in mels_list:
            if self.vocoder_model == "melgan":
                wav = self.vocoder.inverse(mels.transpose(1, 0).unsqueeze(0) / np.log(10))[0]
            elif self.vocoder_model == "hifigan":
                wav = self.vocoder(mels.transpose(1, 0).unsqueeze(0)).squeeze(1)[0]
            else:
                wav = self.vocoder(mels.transpose(1, 0).unsqueeze(0))[0]
            wav = wav.cpu().detach().numpy()
            wavs.append(wav)

        sil = pydub.AudioSegment.silent(300, frame_rate=self.sampling_rate)
        wav_out = sil
        for wav in wavs:
            wav= aukit.anything2bytes(wav, sr=self.sampling_rate)
            wav = pydub.AudioSegment(wav)
            wav_out = wav_out + wav + sil
        wav_out.export(save_file, format="wav")


if __name__ == "__main__":
    synthesizer = Synthesizer("data/checkpoint")
    
    texts = ["我很想你呢，你在干什么？有没有想我？", "今天天气很好你打算做什么？我们一起出去玩吧",
             "你我相识本是一场误会让我们从此别再见", "扶老奶奶过马路使我们都应该做的", "图啥自强自立，不过三十而已",
             "我们的价值观不一样，不能成为好朋友"]
    
    for speaker in ["SSB0005", "SSB0080", "SSB0241", "chenyixun", "lijian", "meizi1", "meizi2"]:
        for i, text in enumerate(texts):
            synthesizer.synthesize(text, speaker, f"demo/demo_{speaker}_{i}.wav")
