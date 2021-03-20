#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Dialobot. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from paws_data import PAWS_X
from typing import Dict, Tuple, List
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pytorch_lightning as pl
import sys
sys.path.append("../../../")


class BartForSeq2SeqLM(pl.LightningModule):
    def __init__(self, src_lang, tgt_lang):
        super().__init__()
        self.batch_size = 16
        self.lr = 3e-5
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-en-ro")

    def forward(self, batch):
        model_inputs, labels = batch
        out = self.model(**model_inputs, labels=labels)
        return out

    def training_step(self, batch, batch_idx):
        """Training steps"""
        out = self.forward(batch)
        loss = out['loss']
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> Dict:
        """Validation steps"""
        out = self.forward(batch)
        loss = out['loss']
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        while True:
            user_input = input()

            if user_input == "stop":
                break
            else:
                inputs = self.tokenizer(user_input, return_tensors="pt")
                translated_tokens = model.generate(
                    **inputs,
                    decoder_start_token_id=self.tokenizer.lang_code_to_id[
                        self.tgt_lang])
                print(
                    self.tokenizer.batch_decode(translated_tokens,
                                                skip_special_tokens=True)[0])

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LambdaLR]]:
        """
        Configure optimizers and lr schedulers

        Returns:
            (Tuple[List[Optimizer], List[LambdaLR]]): [optimizers], [schedulers]
        """

        optimizer = AdamW([p for p in self.parameters() if p.requires_grad],
                          lr=self.lr)

        return {"optimizer": optimizer}

    def train_dataloader(self):
        return DataLoader(
            PAWS_X("../../models/data/x-final/ko/translated_train.tsv",
                   "ko_KR", "ko_KR", 1024),
            batch_size=2,
            pin_memory=True,
            num_workers=16,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            PAWS_X("../../models/data/x-final/ko/dev_2k.tsv", "ko_KR", "ko_KR",
                   1024),
            num_workers=16,
            batch_size=2,
        )


if __name__ == "__main__":
    #trainer = pl.Trainer(gpus=None)
    trainer = pl.Trainer(gpus=-1, auto_select_gpus=True, accelerator="ddp")
    model = BartForSeq2SeqLM("ko_KR", "ko_KR")
    trainer.fit(model)
