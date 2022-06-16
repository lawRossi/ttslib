import pickle
from ttslib.text.symbols import shengmu, yunmu
import json
import lmdb
from ttslib.audio_processing import inv_mel_spec, TacotronSTFT
from scipy.io.wavfile import write
import torch
import struct
import random


# with open("freq.json") as fi:
#     frequences = json.load(fi)

# valid_symbols = set(shengmu + yunmu)

# for token in frequences:
#     assert token in valid_symbols

# for symbol in valid_symbols:
#     frequence = frequences.get(symbol, 0)
#     if frequence < 3:
#         print(symbol, frequence)


# import glob
# import os


# for file in glob.glob("/data/speech/wav1/**", recursive=True):
#     if file.endswith(".wav"):
#         lab_file = file.replace(".wav", ".lab")
#         if os.path.exists(lab_file):
#             print(file)


# with lmdb.open("data/train", map_size=int(1e11)) as env:
#     stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, None)
#     with env.begin(write=False) as txn:
#         for w, v in t
    # with env.begin(write=False) as txn:
    #     keys = []
    #     min_pitch = 1000
    #     max_pitch = -100
    #     min_energy = 1000
    #     max_energy = -100
    #     for k, v in txn.cursor():
    #         keys.append(k)
    #         v = pickle.loads(v)
    #         pitch = v["pitch"]
    #         min_pitch = min(min_pitch, pitch.min())
    #         max_pitch = max(max_pitch, pitch.max())
    #         energy = v["energy"]
    #         min_energy = min(min_energy, energy.min())
    #         max_energy = max(max_energy, energy.max())
    #     print(f"pitch min: {min_pitch} max: {max_pitch}")
    #     print(f"energy min: {min_energy} max: {max_energy}")

import os
import shutil


with open("/data/speech/aishell/metadata.txt") as fi:
    wavs = []
    for line in fi:
        wavs.append(line.strip().split("\t")[0])
    
    random.shuffle(wavs)
    os.makedirs("/data/output/wavs", exist_ok=True)
    
    for i in range(50):
        shutil.copy(os.path.join("/data/speech/aishell", wavs[i]), f"/data/output/wavs/{i}.wav")
