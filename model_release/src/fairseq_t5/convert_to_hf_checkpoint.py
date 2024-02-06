import argparse
import os

import torch

from .configuration_fairseq_t5 import FairseqT5Config
from .modeling_fairseq_t5 import FairseqT5ForConditionalGeneration
from .tokenization_ast_t5 import ASTT5Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str)
    parser.add_argument("--dict_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--hub_model_name", type=str)
    args = parser.parse_args()

    print(f"Loading fairseq state dictionary")
    fairseq_ckpt = torch.load(args.in_path, map_location=torch.device('cpu'))
    fairseq_cfg = fairseq_ckpt['cfg']
    fairseq_model = fairseq_ckpt['model']

    print(f"Loading tokenizer")
    tokenizer = ASTT5Tokenizer(
        args.dict_path,
        n_sentinel_tokens=min(fairseq_cfg['task']['tokens_per_sample'], 1000)
    )

    print('New hf model')
    new_t5config = FairseqT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=fairseq_cfg['model']['encoder_embed_dim'],
        d_kv=fairseq_cfg['model']['encoder_embed_dim'] // fairseq_cfg['model']['encoder_attention_heads'],
        d_ff=fairseq_cfg['model']['encoder_ffn_embed_dim'],
        num_layers=fairseq_cfg['model']['encoder_layers'],
        num_decoder_layers=fairseq_cfg['model']['decoder_layers'],
        num_heads=fairseq_cfg['model']['encoder_attention_heads'],
        relative_attention_num_buckets=fairseq_cfg['model']['encoder_rp_bins'],
        relative_attention_max_distance=fairseq_cfg['model']['encoder_rp_max_dist'],
        max_positions=fairseq_cfg['model']['max_positions'],
        dropout_rate=fairseq_cfg['model']['dropout'],
        layer_norm_epsilon=1e-5,
        initializer_factor=1.0,
        feed_forward_proj=fairseq_cfg['model']['activation_fn'],
        tie_word_embeddings=fairseq_cfg['model']['share_all_embeddings'],
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        torch_dtype="float32"
    )

    new_t5_model = FairseqT5ForConditionalGeneration(new_t5config)

    print(f"Mapping fairseq state dict to hf state dict")
    print("Detected", len(fairseq_model.keys()), "parameter Tensors")

    new_state_dict = {}

    for k, v in fairseq_model.items():
        v = v.half()

        k_out = ''

        if k.startswith("encoder"):
            if k == "encoder.embed_tokens.weight":
                # k_out = 'shared.weight'
                k_out = 'encoder.embed_tokens.weight'

            if k == 'encoder.embed_positions.weight':
                k_out = 'encoder.pos_embed.weight'

            if k == 'encoder.layernorm_embedding.weight':
                k_out = 'encoder.first_layer_norm.weight'
            if k == 'encoder.layernorm_embedding.bias':
                k_out = 'encoder.first_layer_norm.bias'

            if "layers" in k:
                layer_num = k.split('.')[2]
                # print(layer_num)
                if "self_attn.q_proj.weight" in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.q.weight'
                if 'self_attn.q_proj.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.q.bias'

                if "self_attn.k_proj.weight" in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.k.weight'
                if 'self_attn.k_proj.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.k.bias'

                if "self_attn.v_proj.weight" in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.v.weight'
                if 'self_attn.v_proj.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.v.bias'

                if "self_attn.out_proj.weight" in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.o.weight'
                if 'self_attn.out_proj.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.o.bias'

                if 'self_attn_layer_norm.weight' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.layer_norm.weight'
                if 'self_attn_layer_norm.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.layer_norm.bias'

                if 'self_attn_rel_pos' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.0.SelfAttention.relative_attention_bias.weight'

                if 'fc1.weight' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.DenseReluDense.wi.weight'
                if 'fc1.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.DenseReluDense.wi.bias'

                if 'fc2.weight' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.DenseReluDense.wo.weight'
                if 'fc2.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.DenseReluDense.wo.bias'

                if 'final_layer_norm.weight' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.layer_norm.weight'
                if 'final_layer_norm.bias' in k:
                    k_out = 'encoder.block.' + layer_num + '.layer.1.layer_norm.bias'

        if k.startswith("decoder"):
            if k == "decoder.embed_tokens.weight":
                k_out = 'decoder.embed_tokens.weight'

            if k == 'decoder.embed_positions.weight':
                k_out = 'decoder.pos_embed.weight'

            if k == 'decoder.layernorm_embedding.weight':
                k_out = 'decoder.first_layer_norm.weight'
            if k == 'decoder.layernorm_embedding.bias':
                k_out = 'decoder.first_layer_norm.bias'

            if "layers" in k:
                layer_num = k.split('.')[2]

                if "self_attn.q_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.q.weight'
                if 'self_attn.q_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.q.bias'

                if "self_attn.k_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.k.weight'
                if 'self_attn.k_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.k.bias'

                if "self_attn.v_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.v.weight'
                if 'self_attn.v_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.v.bias'

                if "self_attn.out_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.o.weight'
                if 'self_attn.out_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.SelfAttention.o.bias'

                if 'self_attn_layer_norm.weight' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.layer_norm.weight'
                if 'self_attn_layer_norm.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.0.layer_norm.bias'

                if "encoder_attn.q_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.q.weight'
                if 'encoder_attn.q_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.q.bias'

                if "encoder_attn.k_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.k.weight'
                if 'encoder_attn.k_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.k.bias'

                if "encoder_attn.v_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.v.weight'
                if 'encoder_attn.v_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.v.bias'

                if "encoder_attn.out_proj.weight" in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.o.weight'
                if 'encoder_attn.out_proj.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.EncDecAttention.o.bias'

                if 'encoder_attn_layer_norm.weight' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.layer_norm.weight'
                if 'encoder_attn_layer_norm.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.1.layer_norm.bias'

                if 'fc1.weight' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.DenseReluDense.wi.weight'
                if 'fc1.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.DenseReluDense.wi.bias'

                if 'fc2.weight' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.DenseReluDense.wo.weight'
                if 'fc2.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.DenseReluDense.wo.bias'

                if 'final_layer_norm.weight' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.layer_norm.weight'
                if 'final_layer_norm.bias' in k:
                    k_out = 'decoder.block.' + layer_num + '.layer.2.layer_norm.bias'

        if 'decoder.output_projection.weight' in k:
            k_out = 'lm_head.weight'

        if k_out == '':
            print('Parameter not matched:', k)
        else:
            if new_t5_model.state_dict()[k_out].shape != v.shape:
                print(k, k_out, new_t5_model.state_dict()[k_out].shape, v.shape)

            new_state_dict[k_out] = v

            if k == "encoder.embed_tokens.weight":
                k_out_2 = 'shared.weight'
                if new_t5_model.state_dict()[k_out_2].shape != v.shape:
                    print(k, k_out_2, new_t5_model.state_dict()[k_out_2].shape, v.shape)

                new_state_dict[k_out_2] = v

    for k in new_t5_model.state_dict():
        if k not in new_state_dict:
            print('Parameter with no values:', k)

    print('Saving t5 model')
    os.makedirs(args.out_path, exist_ok=True)
    new_t5_model.load_state_dict(new_state_dict)
    new_t5_model.save_pretrained(args.out_path, state_dict=new_state_dict)
    tokenizer.save_pretrained(args.out_path)

    if hasattr(args, "hub_model_name") and args.hub_model_name is not None:
        FairseqT5Config.register_for_auto_class()
        new_t5_model.register_for_auto_class("AutoModelForSeq2SeqLM")
        tokenizer.register_for_auto_class("AutoTokenizer")
        new_t5_model.push_to_hub(args.hub_model_name)
        tokenizer.push_to_hub(args.hub_model_name)


if __name__ == '__main__':
    main()
