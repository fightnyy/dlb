#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import MBartTokenizerFast, MBartModel

import torch
import torch.nn as nn
import pandas as pd


class PAWS_X(torch.utils.data.Dataset):
    """
    infile: file location with file name,
    src_lang : language of the input, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    tgt_lang : language of the output, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    """
    def __init__(self, infile: str, src_lang: str, tgt_lang: str):

        print("#" * 30)
        print("DataPreprocessing Start......")
        print("#" * 30)
        self.tokenizer = MBartTokenizerFast.from_pretrained(
            "facebook/mbart-large-cc25", src_lang=src_lang, tgt_lang=tgt_lang)
        self.data = pd.read_csv(infile, sep="\t", error_bad_lines=False)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.preprocess_data()

        print("#" * 30)
        print("DataPreprocessing End......")
        print("#" * 30)

    def __len__(self):
        assert len(self.data["sentence1"]) == len(self.data["sentence2"])
        return len(self.data)

    def __getitem__(self, idx: int):
        batch: dict = self.tokenizer.prepare_seq2seq_batch(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            padding="max_length",
            src_texts=self.data.iloc[idx]["sentence1"],
            tgt_texts=self.data.iloc[idx]["sentence2"],
            return_tensors="pt",
        )
        tgt_attention_mask = self.make_output_attention(
            batch["labels"].flatten())
        batch.to("cuda:1")
        return (
            batch["input_ids"].flatten().long(),
            batch["labels"].flatten().long(),
            batch["attention_mask"].flatten().long(),
            tgt_attention_mask.long(),
        )

    def preprocess_data(self):
        self.data.drop_duplicates(subset=["sentence1", "sentence2"],
                                  inplace=True)
        self.data.dropna()
        is_paraphrase = self.data['label'] == 1.0
        self.data = self.data[is_paraphrase]

    def make_output_attention(self, tgt_texts):

        one = torch.ones(self.tokenizer.model_max_length)
        zero = torch.zeros(self.tokenizer.model_max_length)
        tgt_attention_mask = torch.where(tgt_texts == torch.tensor(1), zero,
                                         one)

        return tgt_attention_mask
