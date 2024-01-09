import math

import torch.nn as nn

from fairseq.modules import RelativePositionalEmbedding
from fairseq.models import register_model_architecture
from fairseq.utils import safe_getattr
from .transformer_legacy import base_architecture


def init_t5_params(module):
    def normal_(data, scale):
        data.copy_(data.cpu().normal_(mean=0.0, std=scale).to(data.device))

    def uniform_(data, scale):
        data.copy_(data.cpu().uniform_(-math.sqrt(3.0) * scale, math.sqrt(3.0) * scale).to(data.device))

    if isinstance(module, nn.Linear):
        fan_out = module.weight.data.size(0)
        fan_in = module.weight.data.size(0)
        normal_(module.weight.data, (fan_in + fan_out) ** -0.5)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        embed_dim = module.weight.data.size(1)
        uniform_(module.weight.data, embed_dim ** -0.5 / math.sqrt(2))
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, RelativePositionalEmbedding):
        module.rel_attn_bias.weight.data.zero_()


def transformer_t5_common(args):
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = safe_getattr(args, "decoder_normalize_before", False)
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "relu")
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.share_decoder_input_output_embed = safe_getattr(args, "share_decoder_input_output_embed", True)
    args.share_all_embeddings = safe_getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = safe_getattr(args, "no_token_positional_embeddings", False)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.apply_t5_init = safe_getattr(args, "apply_t5_init", True)
    args.disable_flash_attn = safe_getattr(args, "disable_flash_attn", False)
    return base_architecture(args)


@register_model_architecture("transformer", "transformer_t5_small")
def transformer_t5_small(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.decoder_output_dim = safe_getattr(args, "decoder_output_dim", 512)
    args.decoder_input_dim = safe_getattr(args, "decoder_input_dim", 512)
    return transformer_t5_common(args)


@register_model_architecture("transformer", "transformer_t5_base")
def transformer_t5_base(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_output_dim = safe_getattr(args, "decoder_output_dim", 768)
    args.decoder_input_dim = safe_getattr(args, "decoder_input_dim", 768)
    return transformer_t5_common(args)


@register_model_architecture("transformer", "transformer_t5_large")
def transformer_t5_large(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.decoder_output_dim = safe_getattr(args, "decoder_output_dim", 1024)
    args.decoder_input_dim = safe_getattr(args, "decoder_input_dim", 1024)
    return transformer_t5_common(args)


@register_model_architecture("transformer", "transformer_t5_base_rp")
def transformer_t5_base_rp(args):
    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", True)
    args.encoder_rp_bins = safe_getattr(args, "encoder_rp_bins", 32)
    args.encoder_rp_max_dist = safe_getattr(args, "encoder_rp_max_dist", 128)
    args.decoder_rel_pos = safe_getattr(args, "decoder_rel_pos", True)
    args.decoder_rp_bins = safe_getattr(args, "decoder_rp_bins", 32)
    args.decoder_rp_max_dist = safe_getattr(args, "decoder_rp_max_dist", 128)
    return transformer_t5_base(args)


@register_model_architecture("transformer", "transformer_t5_base_rpe")
def transformer_t5_base_rpe(args):
    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", True)
    args.encoder_rp_bins = safe_getattr(args, "encoder_rp_bins", 32)
    args.encoder_rp_max_dist = safe_getattr(args, "encoder_rp_max_dist", 128)
    return transformer_t5_base(args)


@register_model_architecture("transformer", "transformer_t5_large_rpe")
def transformer_t5_large_rpe(args):
    args.encoder_rel_pos = safe_getattr(args, "encoder_rel_pos", True)
    args.encoder_rp_bins = safe_getattr(args, "encoder_rp_bins", 128)
    args.encoder_rp_max_dist = safe_getattr(args, "encoder_rp_max_dist", 256)
    return transformer_t5_large(args)
