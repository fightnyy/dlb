#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import MBart50TokenizerFast

import torch

import pandas as pd


class PAWS_X(torch.utils.data.Dataset):
    """
    infile: file location with file name,
    src_lang : language of the input, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    tgt_lang : language of the output, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    """

    def __init__(self, infile: str, src_lang: str, tgt_lang: str):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-en-ro", src_lang=src_lang,tgt_lang=tgt_lang)
        self.data = pd.read_csv(infile, sep="\t", error_bad_lines=False)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.preprocess_data()


    def __len__(self):
        assert len(self.data["sentence1"]) == len(self.data["sentence2"])
        return len(self.data)

    def __getitem__(self, idx: int):
        model_inputs = self.tokenizer(self.data.iloc[idx]['sentence1'], return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(self.data.iloc[idx]['sentence2'], return_tensors="pt").input_ids
        
        model_input, labels= self.make_pad(model_inputs, labels)

        return model_inputs, labels

    def preprocess_data(self):
        self.data.drop_duplicates(subset=["sentence1", "sentence2"],
                                  inplace=True)
        self.data.dropna()
        is_paraphrase = self.data['label']==1.0
        self.data = self.data[is_paraphrase]

            

    def make_pad(self, tgt_texts):

        one = torch.ones(self.tokenizer.model_max_length)
        zero = torch.zeros(self.tokenizer.model_max_length)
        tgt_attention_mask = torch.where(tgt_texts == torch.tensor(1), zero,
                                         one)

        return tgt_attention_mask 
