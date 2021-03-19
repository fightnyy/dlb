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

import torch
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
from gym.lightning_base import LightningBase


class BartForSeq2SeqLM(LightningBase):
    def __init__(self, cfg_path, cfg_name):
        """
        Constructor of BartForSeq2SeqLM

        Args:
            cfg_path (str): parents path
            cfg_name (str): config file name

        """
        super().__init__(**self.load_args(cfg_path, cfg_name))
        self.model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-cc25")

        if self.precision == 16:
            self.model = self.model.half()

    def _evaluate(self, batch, batch_idx, stage=None):
        input_ids, decoder_input_ids, input_attn, decoder_input_attn = batch
        outputs = model(input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=input_attn,
                        decoder_attention_mask=decoder_input_attn)
        lm_logits = outputs.last_hidden_state
        loss_fn = torch.nn.CrossEntropyLoss()
        import pdb
        pdb.set_trace()

    def training_step(self, batch, batch_idx):
        """Training steps"""
        _evaluate(self, batch, batch_idx, stage=None)
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Validation steps"""
        pass
