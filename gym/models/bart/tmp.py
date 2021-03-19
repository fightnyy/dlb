#!/usr/bin/env python
# -*- coding: utf-8 -*-

from paws_data import PAWS_X
from torch.utils.data import DataLoader
from pdb import set_trace
from transformers import MBartForConditionalGeneration
model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-en-ro")
model = model.to("cuda:1")
for val in DataLoader(PAWS_X("../data/x-final/ko/translated_train.tsv",
                             "ko_KR", "ko_KR"),
                      batch_size=32):
    model(input_ids=val[0],
          attention_mask=val[2],
          decoder_input_ids=val[1],
          decoder_attention_mask=val[3])
    set_trace()
