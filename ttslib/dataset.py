"""
@Author: Rossi
Created At: 2022-04-16
"""

from collections import OrderedDict
import glob
import json
import os.path
from pathlib import Path
import random

import numpy as np
import pickle
import torch
from torchtext.data import batch, Dataset, Example, Field, NestedField, Iterator


class TTSDataset(Dataset):

    def __init__(self, samples, data_dir, fields):
        super().__init__([], fields.values())
        self.samples = samples
        self.data_dir = data_dir
        self.raw_fields = fields

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                value = getattr(example, attr)
                if hasattr(value, "__len__"):
                    return len(value)
        return 0

    def __getitem__(self, i):
        sample = self.samples[i]
        speaker = sample["speaker"]
        audio_file = Path(sample["audio_file"]).name
        mel_file = f"{self.data_dir}/{speaker}_{audio_file.replace('.wav', '.npy')}"
        mels = np.load(mel_file)
        values = {
            "speakers": sample["speaker"],
            "durations": sample["durations"],
            "phonemes": sample["phonemes"],
            "mels": mels
        }
        example = Example.fromdict(values, self.raw_fields)

        return example

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for i in range(len(self.samples)):
            yield self[i]

    def __getattr__(self, attr):
        if attr == "phonemes":
            for sample in self.samples:
                yield sample["phonemes"]   
        elif attr == "speakers":
            for sample in self.samples:
                yield sample["speaker"]
        elif attr in self.raw_fields:
            for example in self:
                yield getattr(example, attr)


class AlignDataset(TTSDataset):

    def __getitem__(self, i):
        sample = self.samples[i]
        speaker = sample["speaker"]
        audio_file = Path(sample["audio_file"]).name
        mel_file = f"{self.data_dir}/{speaker}_{audio_file.replace('.wav', '.npy')}"
        mels = np.load(mel_file)

        blank = "<SP>"
        phonemes_with_blank = [blank]
        for phoneme in sample["phonemes"]:
            phonemes_with_blank.append(phoneme)
            phonemes_with_blank.append(blank)

        values = {
            "mels": mels,
            "phonemes": sample["phonemes"],
            "phonemes_with_blank": phonemes_with_blank,
            "mel_lens": mels.shape[0],
            "phoneme_lens": len(sample["phonemes"])
        }
        example = Example.fromdict(values, self.raw_fields)

        return example


class AdaptationDataset(Dataset):
    def __init__(self, data_dir, fields):
        examples = []
        for file in glob.glob(data_dir+"/*"):
            with open(file, "rb") as fi:
                example = pickle.load(fi)
                example["mels"] = example["mels"].transpose(1, 0)
                examples.append(Example.fromdict(example, fields))
        super().__init__(examples, fields.values())


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """
    def data(self):
        example_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            self.random_shuffler(example_idxs)
        return example_idxs

    def create_batches(self):
        self.batches = self.pool(
            self.batch_size,
            self.sort_key, self.batch_size_fn,
            random_shuffler=self.random_shuffler,
            shuffle=self.shuffle,
            sort_within_batch=self.sort_within_batch
        )

    def pool(self, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
             random_shuffler=None, shuffle=False, sort_within_batch=False):
        """Sort within buckets, then batch, then shuffle batches.

        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        if random_shuffler is None:
            random_shuffler = random.shuffle
        for pack in batch(self.data(), batch_size * 50, batch_size_fn):
            pack = [self.dataset[idx] for idx in pack]
            p_batch = batch(sorted(pack, key=key), batch_size, batch_size_fn) \
                if sort_within_batch \
                else batch(pack, batch_size, batch_size_fn)
            if shuffle:
                for b in random_shuffler(list(p_batch)):
                    yield b
            else:
                for b in list(p_batch):
                    yield b


def tokenize(text):
    return text.split(" ")


def tokenize_int(text):
    return [int(split) for split in text.split(" ")]


def tokenize_float(text):
    return [float(split) for split in text.split(" ")]


def load_tts_dataset(metadata_file, data_dir, use_pitch=True):
    phonemes = Field(batch_first=True, unk_token=None)
    speakers = Field(sequential=False, unk_token=None)
    mel_nested = Field(use_vocab=False, batch_first=True, pad_token=0.0, dtype=torch.float)
    mels = NestedField(mel_nested, use_vocab=False)
    durations = Field(use_vocab=False, pad_token=0, batch_first=True, dtype=torch.int32)

    fields = OrderedDict({
        "phonemes": ("phonemes", phonemes),
        "speakers": ("speakers", speakers),
        "mels": ("mels", mels),
        "durations": ("durations", durations)
    })

    if use_pitch:
        pitch = Field(tokenize=tokenize_float, use_vocab=False, pad_token=0, batch_first=True, dtype=torch.float)
        fields["pitch"] = ("pitch", pitch)

    with open(metadata_file) as fi:
        samples = []
        for line in fi:
            sample = json.loads(line)
            speaker = sample["speaker"]
            audio_file = Path(sample["audio_file"]).name
            mel_file = f"{data_dir}/{speaker}_{audio_file.replace('.wav', '.npy')}"
            if os.path.exists(mel_file):
                samples.append(sample)
    num_eval = int(len(samples) * 0.01)
    train_samples = samples[:-num_eval]
    eval_samples = samples[-num_eval:]
    train_dataset = TTSDataset(train_samples, data_dir, fields)
    eval_dataset = TTSDataset(eval_samples, data_dir, fields)

    phonemes.build_vocab(train_dataset)
    speakers.build_vocab(train_dataset)

    return train_dataset, eval_dataset


def load_align_dataset(metadata_file, data_dir):
    phonemes = Field(tokenize, batch_first=True, unk_token="<SP>")
    mel_nested = Field(use_vocab=False, batch_first=True, pad_token=0.0, dtype=torch.float)
    mels = NestedField(mel_nested, use_vocab=False)
    len_field = Field(sequential=False, use_vocab=False, dtype=torch.int)

    fields = OrderedDict({
        "phonemes": ("phonemes", phonemes),
        "phonemes_with_blank": ("phonemes_with_blank", phonemes),
        "mels": ("mels", mels),
        "phoneme_lens": ("phoneme_lens", len_field),
        "mel_lens": ("mel_lens", len_field)
    })

    with open(metadata_file) as fi:
        samples = [json.loads(line) for line in fi]
    num_eval = int(len(samples) * 0.01)
    train_samples = samples[:-num_eval]
    eval_samples = samples[-num_eval:]
    train_dataset = AlignDataset(train_samples, data_dir, fields)
    eval_dataset = AlignDataset(eval_samples, data_dir, fields)

    phonemes.build_vocab(train_dataset)

    return train_dataset, eval_dataset


def load_adaptation_dataset(data_dir, checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "phonemes.pkl"), "rb") as fi:
        phonemes = pickle.load(fi)

    durations = Field(use_vocab=False, pad_token=0, batch_first=True, dtype=torch.int32)
    mel_nested = Field(use_vocab=False, batch_first=True, pad_token=0.0, dtype=torch.float)
    mels = NestedField(mel_nested, use_vocab=False)
    fields = OrderedDict({
        "phonemes": ("phonemes", phonemes),
        "durations": ("durations", durations),
        "mels": ("mels", mels)
    })

    train_data = AdaptationDataset(data_dir, fields)

    return train_data, train_data


if __name__ == "__main__":
    train_data, eval_data = load_tts_dataset("data/metadata.json", "data/mels", False)
    # train_data, eval_data = load_dataset("data", use_pitch=False)
    # data_iter = BucketIterator(train_data, 32, shuffle=True, sort_key="mels")
    # print(len(train_data), len(eval_data), len(train_data)+len(eval_data))
    # for b in data_iter:
    #     print(b.mels.shape)
    #     print(b.pitch.shape)

    data_iter = BucketIterator(train_data, 2, train=False, sort=False)
    for b in data_iter:
        print(b.mels.shape)
        print(b.phonemes)
        break
