"""
@Author: Rossi
Created At: 2022-04-16
"""

from collections import OrderedDict
from math import floor
import os.path
import random
import struct

import lmdb
import numpy as np
import pickle
import torch
from torchtext.data import batch, Dataset, Example, Field, NestedField, Iterator


class TTSDataset(Dataset):

    def __init__(self, data_dir, fields, speaker_vector_file=None):
        super().__init__([], fields.values())
        env = lmdb.open(data_dir, map_size=int(1e11))
        self.txn = env.begin(write=False)
        self.num_examples = self.txn.stat()["entries"]
        self.raw_fields = fields
        if speaker_vector_file is not None:
            self.speaker_vectors = self._load_speaker_vectors(speaker_vector_file)
        else:
            self.speaker_vectors = None

    def _load_speaker_vectors(self, speaker_vector_file):
        with open(speaker_vector_file, "rb") as fi:
            speaker_vectors = pickle.load(fi)
            for speaker, vectors in speaker_vectors.items():
                speaker_vectors[speaker] = np.array(vectors).mean(axis=0)

            return speaker_vectors

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
        values = pickle.loads(self.txn.get(struct.pack(">I", i)))
        if self.speaker_vectors is not None:
            values["speakers"] = self.speaker_vectors[values["speaker"]]
        else:
            values["speakers"] = values["speaker"]
        example = Example.fromdict(values, self.raw_fields)

        return example

    def _adjust_duration(self, values):
        durations = values["durations"]
        mean_duration = floor(durations.mean())
        new_durations = [mean_duration] * len(durations)
        for i in range(sum(durations) - sum(new_durations)):
            new_durations[i] += 1
        return np.array(new_durations)

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        for i in range(self.num_examples):
            yield self[i]

    def __getattr__(self, attr):
        if attr in self.raw_fields:
            for example in self:
                yield getattr(example, attr)


class AdaptationDataset(TTSDataset):
    def __init__(self, path, fields, speaker_vector_file, speakers):
        super().__init__(path, fields, speaker_vector_file)

        self.idxes = []

        for idx, values in self.txn.cursor():
            values = pickle.loads(values)
            if values["speaker"] in speakers:
                self.idxes.append(idx)

        self.num_examples = len(self.idxes)

    def __getitem__(self, i):
        values = pickle.loads(self.txn.get(self.idxes[i]))
        if self.speaker_vectors is not None:
            values["speakers"] = self.speaker_vectors[values["speaker"]]
        else:
            values["speakers"] = values["speaker"]
        example = Example.fromdict(values, self.raw_fields)
        return example


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
        self.batches = self.pool(self.batch_size,
            self.sort_key, self.batch_size_fn,
            random_shuffler=self.random_shuffler,
            shuffle=self.shuffle,
            sort_within_batch=self.sort_within_batch)

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


def load_dataset(cache_dir, speaker_vector_file=None, use_pitch=True):
    phonemes = Field(tokenize=tokenize, batch_first=True, unk_token=None)
    if speaker_vector_file is None:
        speakers = Field(sequential=False, unk_token=None)
    else:
        speakers = Field(use_vocab=False, batch_first=True, dtype=torch.float)
    mel_nested = Field(use_vocab=False, batch_first=True, pad_token=0.0, dtype=torch.float)
    mels = NestedField(mel_nested, use_vocab=False)
    durations = Field(tokenize=tokenize_int, use_vocab=False, pad_token=0, batch_first=True, dtype=torch.int32)

    fields = OrderedDict({
        "phonemes": ("phonemes", phonemes),
        "speakers": ("speakers", speakers),
        "mels": ("mels", mels),
        "durations": ("durations", durations)
    })
    
    if use_pitch:
        pitch = Field(tokenize=tokenize_float, use_vocab=False, pad_token=0, batch_first=True, dtype=torch.float)
        fields["pitch"] = ("pitch", pitch)

    train_dataset = TTSDataset(os.path.join(cache_dir, "train"), fields, speaker_vector_file)
    eval_dataset = TTSDataset(os.path.join(cache_dir, "eval"), fields, speaker_vector_file)

    phonemes.build_vocab(train_dataset)

    if speaker_vector_file is None:
        speakers.build_vocab(train_dataset)
    print("phonemes vocab size: ", len(phonemes.vocab))

    return train_dataset, eval_dataset


def load_adaptation_dataset(cache_dir, checkpoint_dir, speaker_vector_file, speaker_list):
    with open(os.path.join(checkpoint_dir, "phonemes.pkl"), "rb") as fi:
        phonemes = pickle.load(fi)

    pitch = Field(tokenize=tokenize_float, use_vocab=False, pad_token=0, batch_first=True, dtype=torch.float)
    durations = Field(tokenize=tokenize_int, use_vocab=False, pad_token=0, batch_first=True, dtype=torch.int32)
    if speaker_vector_file is None:
        speakers = Field(sequential=False, unk_token=None)
    else:
        speakers = Field(use_vocab=False, batch_first=True, dtype=torch.float)
    mel_nested = Field(use_vocab=False, batch_first=True, pad_token=0.0, dtype=torch.float)
    mels = NestedField(mel_nested, use_vocab=False)
    fields = OrderedDict({
        "phonemes": ("phonemes", phonemes),
        "pitch": ("pitch", pitch),
        "durations": ("durations", durations),
        "speakers": ("speakers", speakers),
        "mels": ("mels", mels)
    })
    
    train_dir = os.path.join(cache_dir, "train")
    eval_dir = os.path.join(cache_dir, "eval")
    train_data = AdaptationDataset(train_dir, fields, speaker_vector_file, speaker_list)
    eval_data = TTSDataset(eval_dir, fields, speaker_vector_file)

    return train_data, eval_data


if __name__ == "__main__":
    # train_data, eval_data = load_adaptation_dataset("data", "data/checkpoint", "data/speakers.pkl", ["SSB0005", "SSB0080"])

    train_data, eval_data = load_dataset("data")
    # data_iter = BucketIterator(train_data, 32, shuffle=True, sort_key="mels")
    # print(len(train_data), len(eval_data), len(train_data)+len(eval_data))
    # for b in data_iter:
    #     print(b.mels.shape)
    #     print(b.pitch.shape)

    data_iter = BucketIterator(eval_data, 2)
    for b in data_iter:
        print(b.phonemes)
        print(b.mels.shape)
        break