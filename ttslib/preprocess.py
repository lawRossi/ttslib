"""
@Author: Rossi
Created At: 2022-04-16
"""

from collections import defaultdict
import json
from math import ceil, floor
from multiprocessing import Process, set_start_method
import os
from pathlib import Path
import random

import librosa
from loguru import logger
import numpy as np
import pickle
import tgt
import torch
import yaml

from ttslib.audio_processing import compute_mel_spectrogram_and_energy, get_mel_from_audio
from ttslib.audio_processing import TacotronSTFT


class Worker(Process):
    def __init__(self, config, samples):
        super().__init__()
        self.stft = TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )
        self.samples = samples
        self.config = config

    def run(self) -> None:
        logger.info(f"start to process {len(self.samples)} samples")
        save_dir = self.config["output_dir"]
        n = 0
        for sample in self.samples:
            max_mel_len = self.config["max_mel_len"]
            audio_file = sample["audio_file"]
            audio, _ = librosa.load(audio_file, sr=None)
            audio = audio[sample["start"]:sample["end"]]
            durations = sample["durations"]
            mel, _ = compute_mel_spectrogram_and_energy(audio, self.stft, durations)
            if mel is None:
                logger.warning(f"invalid mel spectrogram for audio: {audio_file}")
                continue
            if mel.shape[0] > max_mel_len:
                logger.warning("mel too long")
                continue
            filename = Path(sample["audio_file"]).name
            filename = sample["speaker"] + "_" + filename
            save_file = os.path.join(save_dir, filename.replace('.wav', '.npy'))
            np.save(save_file, mel)
            n += 1
            if n % 1000 == 0:
                logger.info(f"{n} samples processed")


def parse_metadata_file(config):
    metadata_file = os.path.join(config["data_dir"], "metadata.txt")
    corpus_dir = config["aligned_corpus_dir"]
    sampling_rate = config["audio"]["sampling_rate"]
    hop_length = config["stft"]["hop_length"]
    max_text_len = config["max_text_len"]
    save_file = config["metadata_file"]

    num_invalid = 0
    samples = []
    with open(metadata_file, encoding="utf-8") as fi, \
            open(save_file, "w", encoding="utf-8") as fo:
        base_dir = os.path.dirname(os.path.abspath(metadata_file))
        for line in fi:
            audio_file, _, speaker = line.strip().split("\t")
            textgrid_file = os.path.join(corpus_dir, "/".join(Path(audio_file).parts[1:]).replace(".wav", ".TextGrid"))
            audio_file = os.path.join(base_dir, audio_file)
            if not os.path.exists(audio_file) or not os.path.exists(textgrid_file):
                continue
            phonemes, durations, start, end = get_alignment(textgrid_file, sampling_rate, hop_length)
            if phonemes is None:
                logger.info(f"invalid : {textgrid_file}")
                num_invalid += 1
                continue
            if len(phonemes) > max_text_len:
                logger.info(f"transcript too long : {textgrid_file}")
                num_invalid += 1
                continue
            if 0 in durations:
                logger.info(f"invalid duration: {textgrid_file}")
                num_invalid += 1
                continue

            samples.append(json.dumps({
                "audio_file": audio_file,
                "speaker": speaker,
                "phonemes": phonemes,
                "durations": durations,
                "start": start,
                "end": end
                }))
        random.shuffle(samples)
        fo.write("\n".join(samples))
        logger.info(f"{num_invalid} invalid files")


def get_alignment(textgrid_file, sampling_rate, hop_length):
    textgrid = tgt.io.read_textgrid(textgrid_file, include_empty_intervals=True)
    tier = textgrid.get_tier_by_name("phones")
    sil_phonemes = ["sil", "sp", "spn", ""]
    phonemes = []
    intervals = []
    end_idx = 0
    max_time = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # trim leading silences
        if phonemes == [] and p in sil_phonemes:
            continue
        max_time = max(max_time, e)
        if p not in sil_phonemes:
            phonemes.append(p)
            end_idx = len(phonemes)
        else:
            if p == "":
                p = "sp"
            phonemes.append(p)
        intervals.append((s, e))
    phonemes = phonemes[:end_idx]  # trim ending silences
    intervals = intervals[:end_idx]
    if not intervals:
        return None, None, None, None

    valid_phonemes = []
    durations = []
    for i, (phoneme, (s, e)) in enumerate(zip(phonemes, intervals)):
        if i == 0:
            s = max(0, s - 0.02)
            start_time = s
        if i == len(intervals) - 1:
            end_time = min(max_time, e + 0.1)
        duration = round(e * sampling_rate / hop_length) - round(s * sampling_rate / hop_length)
        if duration != 0 or phoneme not in sil_phonemes:
            valid_phonemes.append(phoneme)
            durations.append(duration)
    start = floor(start_time * sampling_rate)
    end = ceil(end_time * sampling_rate)
    if sum(durations) > (end - start) // hop_length:
        end += ceil((sum(durations) - (end-start) / hop_length) * hop_length)
    return phonemes, durations, start, end


def load_samples(transcript_file):
    speaker_samples = defaultdict(list)
    with open(transcript_file, encoding="utf-8") as fi:
        for line in fi:
            samples = json.loads(line)
            speaker_samples[samples["speaker"]].append(samples)
    samples = []
    for speaker, sub_samples in speaker_samples.items():
        if len(sub_samples) < 5:
            print(f"less than 5 samples: {speaker}")
            continue
        samples.extend(sub_samples)
    return samples


def preprocess(config):
    samples = load_samples(config["metadata_file"])
    logger.info(f"{len(samples)} samples loaded")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # num_workers = config["num_workers"]
    # num_per_worker = ceil(len(samples) / num_workers)
    # groups = [samples[i*num_per_worker: (i+1)*num_per_worker] for i in range(num_workers)]
    # workers = []
    # for group in groups:
    #     worker = Worker(config, group)
    #     workers.append(worker)
    #     worker.start()
    # for worker in workers:
    #     worker.join()

    stft = TacotronSTFT(
        config["stft"]["filter_length"],
        config["stft"]["hop_length"],
        config["stft"]["win_length"],
        config["mel"]["n_mel_channels"],
        config["audio"]["sampling_rate"],
        config["mel"]["mel_fmin"],
        config["mel"]["mel_fmax"],
    )

    max_mel_len = config["max_mel_len"]
    save_dir = config["output_dir"]
    n = 0
    for sample in samples:
        audio_file = sample["audio_file"]
        audio, _ = librosa.load(audio_file, sr=None)
        audio = audio[sample["start"]:sample["end"]]
        durations = sample["durations"]
        mel, _ = compute_mel_spectrogram_and_energy(audio, stft, durations)
        if mel is None:
            logger.warning(f"invalid mel spectrogram for audio: {audio_file}")
            continue
        if mel.shape[0] > max_mel_len:
            logger.warning("mel too long")
            continue
        filename = Path(sample["audio_file"]).name
        filename = sample["speaker"] + "_" + filename
        save_file = os.path.join(save_dir, filename.replace('.wav', '.npy'))
        np.save(save_file, mel)
        n += 1
        if n % 1000 == 0:
            logger.info(f"{n} samples processed")

    logger.info("finished preprocessing")


def build_references(config, speakers, save_file):
    samples = load_samples(config["metadata_file"])
    samples = [sample for sample in samples if sample["speaker"] in speakers]
    stft = TacotronSTFT(
        config["stft"]["filter_length"],
        config["stft"]["hop_length"],
        config["stft"]["win_length"],
        config["mel"]["n_mel_channels"],
        config["audio"]["sampling_rate"],
        config["mel"]["mel_fmin"],
        config["mel"]["mel_fmax"],
    )

    sampling_rate = config["audio"]["sampling_rate"]
    references = dict()

    for sample in samples:
        durations = sample.get("durations")
        durations = np.array(sample["durations"])

        audio_file = sample["audio_file"]
        audio, _ = librosa.load(audio_file, sr=sampling_rate)
        if "start" in sample:
            audio = audio[sample["start"]:sample["end"]]
        mel_spectrogram, _ = compute_mel_spectrogram_and_energy(audio, stft, durations)
        if mel_spectrogram is not None:
            filename = Path(audio_file).name
            key = sample["speaker"] + ":" + filename
            references[key] = {"mel": mel_spectrogram, "durations": durations}

    with open(save_file, "wb") as fo:
        pickle.dump(references, fo)


def generate_align_data(config, metadata_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    stft = TacotronSTFT(
        config["stft"]["filter_length"],
        config["stft"]["hop_length"],
        config["stft"]["win_length"],
        config["mel"]["n_mel_channels"],
        config["audio"]["sampling_rate"],
        config["mel"]["mel_fmin"],
        config["mel"]["mel_fmax"],
    )

    with open(metadata_file) as fi:
        with torch.no_grad():
            for line in fi:
                sample = json.loads(line)
                if not os.path.exists(sample["audio_file"]):
                    continue
                audio, _ = librosa.load(sample["audio_file"], None)
                mel, _ = get_mel_from_audio(audio, stft)
                filename = Path(sample["audio_file"]).name
                filename = sample["speaker"] + "_" + filename
                save_file = os.path.join(save_dir, filename.replace('.wav', '.npy'))
                np.save(save_file, mel.transpose(1, 0))


if __name__ == "__main__":
    set_start_method("spawn")

    config = yaml.load(open("data/preprocess.yaml"), Loader=yaml.FullLoader)
    # parse_metadata_file(config)
    preprocess(config)

    # build_references(config, ["SSB0005", "SSB0080", "meizi1", "meizi2", "chenyixun", "lijian"], "data/references.pkl")

    # prepare_aligning_speech_alignments(config)

    # dump_speaker_vectors("data/speakers", "data/speakers.json")

    # generate_align_data(config, "data/metadata.json", "data/mels")
