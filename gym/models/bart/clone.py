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

import sys

sys.path.append("../../../")
import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MBartTokenizer
from gym.lightning_base import LightningBase
from typing import Dict, Tuple, List
from paws_data import PAWS_X
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn.functional as F


class BartForSeq2SeqLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.lr = 3e-5
        self.tokenizer = MBartTokenizer.from_pretrained(
            "facebook/mbart-large-cc25")
        self.model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-cc25")

    def forward(self, batch):
        input_ids, decoder_input_ids, input_attn, decoder_attention_mask = batch
        out = self.model(input_ids=input_ids,
                         decoder_input_ids=decoder_input_ids,
                         attention_mask=input_attn,
                         decoder_attention_mask=decoder_attention_mask)
        return out

    def _evaluate(self, batch, batch_idx, stage=None):
        outputs = self.forward(batch)
        import pdb
        pdb.set_trace()
        logits = outputs['logits']
        logits = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, decoder_input_ids)
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Training steps"""
        output = self.forward(batch)
        import pdb
        pdb.set_trace()
        logits = outputs['logits']
        logits = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits,
                               decoder_input_ids,
                               ignore_index=self.tokenizer.all_special_ids)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> Dict:
        """Validation steps"""
        return self._evaluate(batch, batch_idx, stage="valid")

    def test_step(self, batch, batch_idx):
        while True:
            user_input = input()

            if user_input == "stop":
                break
            else:
                break

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
                   "ko_KR", "ko_KR"),
            batch_size=32,
            num_workers=32,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            PAWS_X("../../models/data/x-final/ko/dev_2k.tsv", "ko_KR",
                   "ko_KR"),
            batch_size=32,
            pin_memory=True,
            num_workers=32,
        )


if __name__ == "__main__":
    model = BartForSeq2SeqLM()
    trainer = pl.Trainer(gpus=-1, auto_select_gpus=True, accelerator="ddp")
    trainer.fit(model)
