"""
@Author: Rossi
Created At: 2022-05-28
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# def compute_pairwise_distances(p_encodings, m_encodings, p_lens, m_lens):
#     pairwise_distances = -torch.cdist(m_encodings, p_encodings, p=2)
#     device = p_encodings.device
#     p_mask = torch.arange(p_encodings.shape[1]) >= p_lens.unsqueeze(-1).to(device)
#     m_mask = torch.arange(m_encodings.shape[1]) >= m_lens.unsqueeze(-1).to(device)

#     pairwise_distances = pairwise_distances.masked_fill(m_mask.unsqueeze(-1), -1e9)
#     pairwise_distances = pairwise_distances.masked_fill(p_mask.unsqueeze(1), -1e9)

#     return pairwise_distances


def compute_pairwise_distances(p_encodings, m_encodings, p_lens, m_lens, T=0.05):
    p_encodings = F.normalize(p_encodings, p=2, dim=2)
    m_encodings = F.normalize(m_encodings, p=2, dim=2)
    pairwise_distances = torch.bmm(m_encodings, p_encodings.permute(0, 2, 1)) / T
    device = p_encodings.device
    p_mask = torch.arange(p_encodings.shape[1]) >= p_lens.unsqueeze(-1).to(device)
    m_mask = torch.arange(m_encodings.shape[1]) >= m_lens.unsqueeze(-1).to(device)

    pairwise_distances = pairwise_distances.masked_fill(m_mask.unsqueeze(-1), -1e9)
    pairwise_distances = pairwise_distances.masked_fill(p_mask.unsqueeze(1), -1e9)

    return pairwise_distances


def compute_masks(alignments, p_len, m_len):
    if isinstance(alignments, torch.Tensor):
        alignments = alignments.detach().numpy()
    masks = np.ones_like(alignments)
    width = int(m_len / p_len * 1.5)

    for i in range(p_len):
        start = max(0, math.floor(i * m_len / p_len - width))
        end = min(m_len, math.ceil(i * m_len / p_len + width))
        for j in range(start, end):
            masks[i][j] = 0
    return masks


def get_hard_alignments(log_probs, batch_phonemes, p_lens, m_lens, blank):
    log_probs = log_probs.cpu().detach().numpy()
    hard_alignments = np.zeros_like(log_probs, dtype=np.int)
    log_probs = log_probs.transpose(0, 2, 1)

    dp = np.ndarray((log_probs.shape[1], log_probs.shape[2], 2))

    for idx in range(log_probs.shape[0]):
        phonemes = batch_phonemes[idx].numpy()
        alignment = log_probs[idx]
        masks = compute_masks(alignment, p_lens[idx], m_lens[idx])
        alignment += masks * -1e9

        dp.fill(-np.inf)
        dp[0, :, 0] = np.cumsum(alignment[0])

        for j in range(1, m_lens[idx]):
            for i in range(1, min(j+1, p_lens[idx])):
                if phonemes[i-1] == blank:
                    dp[i, j, 0] = max(dp[i-1, j-1, 0], dp[i-1, j-1, 1], dp[i, j-1, 0]) + alignment[i, j]
                else:
                    dp[i, j, 0] = max(dp[i-1, j-1, 0], dp[i, j-1, 0]) + alignment[i, j]
                if phonemes[i] == blank and i != p_lens[idx] - 1:
                    dp[i, j, 1] = max(dp[i-1, j-1, 0], dp[i, j-1, 0])

        mapping = np.array([0] * m_lens[idx])
        if dp[p_lens[idx]-1, m_lens[idx]-1, 0] > dp[p_lens[idx]-2, m_lens[idx]-1, 0]:
            mapping[-1] = p_lens[idx] - 1
        else:
            mapping[-1] = p_lens[idx] - 2

        for j in range(m_lens[idx]-2, -1, -1):
            i = mapping[j+1]
            if i == 0:
                break
            if dp[i, j, 0] > dp[i-1, j, 0]:
                if dp[i, j, 1] > dp[i, j, 0]:
                    mapping[j] = i - 1
                else:
                    mapping[j] = i
            else:
                if dp[i-1, j, 0] < dp[i-1, j, 1]:
                    mapping[j] = i-2
                else:
                    mapping[j] = i - 1

        for j, i in enumerate(mapping):
            hard_alignments[idx, j, i] = 1

    return torch.tensor(hard_alignments, dtype=torch.float)


def get_phonemes_durations_pitch(phonemes_with_blank, hard_alignments):
    batch_phonemes = phonemes_with_blank.numpy()
    batch_durations = hard_alignments.sum(axis=1).numpy()
    phonemes = []
    durations = []
    max_len = 0
    for i in range(batch_phonemes.shape[0]):
        current_phonemes = []
        current_durations = []
        for phoneme, duration in zip(batch_phonemes[i], batch_durations[i]):
            if duration > 0:
                current_phonemes.append(phoneme)
                current_durations.append(duration)
        phonemes.append(current_phonemes)
        durations.append(current_durations)
        max_len = max(max_len, len(current_phonemes))
    for i in range(len(phonemes)):
        phonemes[i] += [1] * (max_len - len(phonemes[i]))   # pad token of phonemes is 1
        durations[i] += [0] * (max_len - len(durations[i]))
    phonemes = torch.tensor(phonemes, dtype=torch.long)
    durations = torch.tensor(durations, dtype=torch.int)

    return phonemes, durations


if __name__ == "__main__":
    p_enc = torch.rand((3, 5, 256))
    m_enc = torch.rand((3, 8, 256))
    pitch = torch.rand((3, 8))

    p_lens = torch.tensor([3, 4, 5])
    m_lens = torch.tensor([6, 7, 8])

    pairwise_distances = compute_pairwise_distances(p_enc, m_enc, p_lens, m_lens)

    log_probs = torch.log_softmax(pairwise_distances, dim=2)

    phonemes = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 3, 0], [0, 1, 0, 2, 3]])

    hard_alignments = get_hard_alignments(log_probs, phonemes, p_lens, m_lens, 0)

    print(hard_alignments)

    phonemes, durations, pitch = get_phonemes_durations_pitch(phonemes, hard_alignments, pitch)

    print(phonemes)
    print(durations)
    print(pitch)
