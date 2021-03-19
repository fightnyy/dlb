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

from unittest import TestCase
from omegaconf import OmegaConf, DictConfig
from hydra.experimental import compose, initialize

import hydra


@hydra.main(config_name="../configs/bart_for_paraphrase_generation")
def test_hydra(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


class TestHydra(TestCase):
    """
    Test codes for hydra experimental compose API
    """

    @staticmethod
    def load_args() -> DictConfig:
        initialize(config_path="../configs/")
        return compose(config_name="bart_for_paraphrase_generation")


if __name__ == '__main__':
    test_hydra()

    testcase = TestHydra()
    args = testcase.load_args()
    print(args)
