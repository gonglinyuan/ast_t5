# from COCO-LM
import math

import torch
import torch.nn as nn


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = torch.sign(relative_position)
    num_buckets //= 2
    n = torch.abs(relative_position)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact
    max_bucket_val = num_buckets - 1 - max_exact
    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + torch.ceil(
        torch.log(n.float() / max_exact)
        / math.log((max_distance - 1) / max_exact)
        * max_bucket_val
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret = torch.where(is_small, n, val_if_large) * sign
    return ret


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, bins, max_dist, attention_heads, max_positions):
        super().__init__()
        self.bins = bins
        self.max_dist = max_dist
        self.attention_heads = attention_heads
        self.max_positions = max_positions
        self.rel_attn_bias = nn.Embedding(self.bins, self.attention_heads)
        relative_position = (
            torch.arange(max_positions, dtype=torch.long)[None, :]
            - torch.arange(max_positions, dtype=torch.long)[:, None]
        )
        self.rp_bucket = relative_position_bucket(
            relative_position,
            num_buckets=self.bins,
            max_distance=self.max_dist
        )
        self.rp_bucket -= self.rp_bucket.min()

    def forward(self, query_len, key_len, batch_size, attn_mask=None, incremental_inference=False):
        if self.rp_bucket.device != self.rel_attn_bias.weight.device:
            self.rp_bucket = self.rp_bucket.to(self.rel_attn_bias.weight.device)
        rp_bucket = self.rp_bucket[:query_len, :key_len]  # [query_len, key_len]
        values = self.rel_attn_bias(rp_bucket)  # [query_len, key_len, attn_heads]
        values = values.permute([2, 0, 1]).contiguous()  # [attn_heads, query_len, key_len]
        values = values.repeat(batch_size, 1, 1)  # [attn_heads * batch_size, query_len, key_len]
        if incremental_inference:
            values = values[:, -1:, :]
        if attn_mask is not None:
            values = values + attn_mask
        return values
