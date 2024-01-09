#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
import typing as tp
from argparse import Namespace
from itertools import zip_longest

from fairseq import options, tasks, utils
from fairseq.binarizer import (
    FileBinarizer,
    VocabularyDatasetBinarizer,
)
from fairseq.data import Dictionary

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")

#####################################################################
# file name tools
#####################################################################


def _train_path(lang, trainpref):
    return "{}{}".format(trainpref, ("." + lang) if lang else "")


def _file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def _dest_path(prefix, lang, destdir):
    return os.path.join(destdir, _file_name(prefix, lang))


def _dict_path(lang, destdir):
    return _dest_path("dict", lang, destdir) + ".txt"


def dataset_dest_prefix(args, output_prefix, lang):
    base = os.path.join(args.destdir, output_prefix)
    if lang is not None:
        lang_part = f".{args.source_lang}-{args.target_lang}.{lang}"
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = f".{args.source_lang}-{args.target_lang}"

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    return "{}.{}".format(dataset_dest_prefix(args, output_prefix, lang), extension)


#####################################################################
# dictionary tools
#####################################################################


def _build_dictionary(
    filenames,
    task,
    args,
    src=False,
    tgt=False,
):
    assert src ^ tgt
    return task.build_dictionary(
        filenames,
        workers=args.workers,
        threshold=args.thresholdsrc if src else args.thresholdtgt,
        nwords=args.nwordssrc if src else args.nwordstgt,
        padding_factor=args.padding_factor,
    )


#####################################################################
# bin file creation logic
#####################################################################


def _make_binary_dataset(
    vocab: Dictionary,
    input_prefix: str,
    output_prefix: str,
    lang: tp.Optional[str],
    num_workers: int,
    args: Namespace,
):
    logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))

    binarizer = VocabularyDatasetBinarizer(
        vocab,
        append_eos=True,
    )

    input_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "")
    full_output_prefix = dataset_dest_prefix(args, output_prefix, lang)

    final_summary = FileBinarizer.multiprocess_dataset(
        input_file,
        args.dataset_impl,
        binarizer,
        full_output_prefix,
        vocab_size=len(vocab),
        num_workers=num_workers,
    )

    logger.info(f"[{lang}] {input_file}: {final_summary} (by {vocab.unk_word})")


#####################################################################
# routing logic
#####################################################################


def _make_dataset(
    vocab: Dictionary,
    input_prefix: str,
    output_prefix: str,
    lang: tp.Optional[str],
    args: Namespace,
    num_workers: int,
):
    if args.dataset_impl == "raw":
        # Copy original text file to destination folder
        output_text_file = _dest_path(
            output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
            lang,
            args.destdir,
        )
        shutil.copyfile(_file_name(input_prefix, lang), output_text_file)
    else:
        _make_binary_dataset(
            vocab, input_prefix, output_prefix, lang, num_workers, args
        )


def _make_all(lang, vocab, args):
    if args.trainpref:
        _make_dataset(
            vocab, args.trainpref, "train", lang, args=args, num_workers=args.workers
        )
    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            _make_dataset(
                vocab, validpref, outprefix, lang, args=args, num_workers=args.workers
            )
    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            _make_dataset(
                vocab, testpref, outprefix, lang, args=args, num_workers=args.workers
            )


#####################################################################
# MAIN
#####################################################################


def main(args):
    # setup some basic things
    utils.import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir, "preprocess.log"),
        )
    )
    logger.info(args)

    assert (
        args.dataset_impl != "huffman"
    ), "preprocessing.py doesn't support Huffman yet, use HuffmanCodeBuilder directly."

    # build dictionaries

    target = not args.only_source

    if not args.srcdict and os.path.exists(_dict_path(args.source_lang, args.destdir)):
        raise FileExistsError(_dict_path(args.source_lang, args.destdir))

    if (
        target
        and not args.tgtdict
        and os.path.exists(_dict_path(args.target_lang, args.destdir))
    ):
        raise FileExistsError(_dict_path(args.target_lang, args.destdir))

    task = tasks.get_task(args.task)

    if args.joined_dictionary:
        assert (
            not args.srcdict or not args.tgtdict
        ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = _build_dictionary(
                {
                    _train_path(lang, args.trainpref)
                    for lang in [args.source_lang, args.target_lang]
                },
                task=task,
                args=args,
                src=True,
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = _build_dictionary(
                [_train_path(args.source_lang, args.trainpref)],
                task=task,
                args=args,
                src=True,
            )

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = _build_dictionary(
                    [_train_path(args.target_lang, args.trainpref)],
                    task=task,
                    args=args,
                    tgt=True,
                )
        else:
            tgt_dict = None

    # save dictionaries

    src_dict.save(_dict_path(args.source_lang, args.destdir))
    if target and tgt_dict is not None:
        tgt_dict.save(_dict_path(args.target_lang, args.destdir))

    if args.dict_only:
        return

    _make_all(args.source_lang, src_dict, args)
    if target:
        _make_all(args.target_lang, tgt_dict, args)

    logger.info("Wrote preprocessed data to {}".format(args.destdir))


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
