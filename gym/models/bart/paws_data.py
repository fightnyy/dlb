#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import MBart50TokenizerFast
from typing import Dict, List
from collections import defaultdict
import torch
import pdb

import pandas as pd


class PAWS_X(torch.utils.data.Dataset):
    """
    infile: file location with file name,
    src_lang : language of the input, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    tgt_lang : language of the output, example : English :en_XX, chinese:zh_CN, japanese:ja_XX, korean : ko_KR,
    """

    def __init__(self, infile: str, src_lang: str, tgt_lang: str,
                 max_len: int):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-en-ro", src_lang=src_lang, tgt_lang=tgt_lang)
        self.data = pd.read_csv(infile, sep="\t", error_bad_lines=False)
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.preprocess_data()

    def __len__(self):
        assert len(self.data["sentence1"]) == len(self.data["sentence2"])
        return len(self.data)

    def __getitem__(self, idx: int):
        model_inputs = self.tokenizer(self.data.iloc[idx]['sentence1'],
                                      return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(self.data.iloc[idx]['sentence2'],
                                    return_tensors="pt").input_ids

        input_ids = model_inputs['input_ids'][0].tolist()
      
        pad_len = len(input_ids)
        pad = [self.tokenizer.pad_token_id] * (self.max_len - pad_len)
        input_ids.extend(pad)

        attn = model_inputs['attention_mask'][0].tolist()
        attn_pad = [0] * (self.max_len - pad_len)
        attn.extend(attn_pad)
        label = labels[0].tolist()

        labels_pad = [self.tokenizer.pad_token_id
                      ] * (self.max_len - len(label))
        label.extend(labels_pad)

        # new_model_inputs = defaultdict(list)
        # new_label = []
        model_inputs['input_ids'] = torch.tensor(input_ids).long()
        model_inputs['attention_mask'] = torch.tensor(attn).long()
        label = torch.tensor(label).long()

        return model_inputs, label

    def preprocess_data(self):
        self.data.drop_duplicates(subset=["sentence1", "sentence2"],
                                  inplace=True)
        self.data.dropna()
        is_paraphrase = self.data['label'] == 1.0
        self.data = self.data[is_paraphrase]
        length = len(self.data)
        delete_list = []

        
        for i in range(length):

          inputs = self.tokenizer(self.data.iloc[i]['sentence1'],return_tensors="pt")
          inputs = inputs['input_ids'][0].tolist()

          with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(self.data.iloc[i]['sentence2'], return_tensors="pt").input_ids
          label = labels[0].tolist()
          if len(inputs) > self.max_len :
            delete_list.append(i)
          if len(label) > self.max_len:
            delete_list.append(i)
        for idx, value in enumerate(delete_list):
          index_name=self.data[self.data['id'] == self.data.iloc[value-idx].id].index
          self.data.drop(index_name, inplace=True)


    # def make_padding(self, model_inputs:Dict,labels:torch.Tensor):
    #
    #     model_inputs['']
    #
    #     return tgt_attention_mask
