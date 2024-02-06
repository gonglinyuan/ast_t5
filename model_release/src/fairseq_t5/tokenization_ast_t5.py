import os
import re
from shutil import copyfile
from typing import List, Optional

import tiktoken
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from .fairseq_dictionary import Dictionary

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "dict_path": "dict.txt"
}

GPT4ENC = tiktoken.encoding_for_model("gpt-4")


class GPT4Dictionary:
    def __init__(self):
        self.vocab = {}
        self.words = {}
        self.vocab_cnt = 4
        for i in range(GPT4ENC.n_vocab):
            try:
                w = GPT4ENC.decode_single_token_bytes(i)
                self.vocab[w] = self.vocab_cnt
                self.words[self.vocab_cnt] = w
                self.vocab_cnt += 1
            except KeyError:
                pass
        self.eos_index = 2
        self.words[2] = b"</s>"
        self.sentinel_start = self.vocab_cnt
        for i in range(1000):
            self.words[self.sentinel_start + i] = f"<sen{i:03d}>".encode("utf-8")

    def index(self, w):
        assert w in self.vocab
        return self.vocab[w]

    def __getitem__(self, i):
        if i in self.words:
            return self.words[i]
        else:
            return b""


class ASTT5Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        dict_path,
        n_sentinel_tokens=0,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        **kwargs
    ) -> None:

        self.dict_path = dict_path
        self.tik_dict = GPT4Dictionary()
        self.fs_dict = Dictionary.load(dict_path)
        self.fs_dict_sentinel_start = len(self.fs_dict)
        for i in range(n_sentinel_tokens):
            self.fs_dict.add_symbol(f'<sen{i:03d}>')

        if "sep_token" in kwargs:
            assert kwargs["sep_token"] == eos_token
            kwargs.pop("sep_token")
        if "cls_token" in kwargs:
            assert kwargs["cls_token"] == bos_token
            kwargs.pop("cls_token")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=eos_token,
            cls_token=bos_token,
            n_sentinel_tokens=n_sentinel_tokens,
            **kwargs, )

    @property
    def vocab_size(self):
        return len(self.fs_dict)

    def get_vocab(self):
        return self.fs_dict.indices

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        mask_0 = [(0 if w < self.fs_dict_sentinel_start else 1) for w in token_ids_0]
        mask_1 = [(0 if w < self.fs_dict_sentinel_start else 1) for w in token_ids_1]
        if token_ids_1 is None:
            return mask_0 + [1]
        return mask_0 + [1] + mask_1 + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep + token_ids_1 + sep) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        return token_ids_0 + sep + token_ids_1 + sep

    def _tokenize(self, text: str) -> List[str]:
        parts = re.split(r"(<sen\d+>)", text)
        tokenized = []
        for part in parts:
            if re.match(r"<sen\d+>", part):
                tokenized.append(part)
            else:
                tokenized.extend(
                    [
                        self.fs_dict[self.tik_dict.index(w)]
                        for w in GPT4ENC.decode_tokens_bytes(GPT4ENC.encode_ordinary(part))
                    ]
                )
        return tokenized

    def _convert_token_to_id(self, token):
        return self.fs_dict.index(token)

    def _convert_id_to_token(self, index):
        return self.fs_dict[index]

    def convert_tokens_to_string(self, tokens):
        token_bytes = b"".join([self.tik_dict[self.fs_dict.index(token)] for token in tokens])
        return token_bytes.decode("utf-8", errors="ignore")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_dict_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["dict_path"]
        )

        if os.path.abspath(self.dict_path) != os.path.abspath(out_dict_path):
            copyfile(self.dict_path, out_dict_path)
            logger.info(f"Copy from {self.dict_path} to {out_dict_path}")

        return (out_dict_path,)
