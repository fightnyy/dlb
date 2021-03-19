from gym.models.bart.bart_for_seq2seq_lm import BartForSeq2SeqLM
from torch.utils.data import DataLoader
from gym.models.bart.pasw_data import PAWS_X

model = BartForSeq2SeqLM(
    cfg_path="../configs",
    cfg_name="bart_for_paraphrase_generation",
)

model.fit(
    train_dataloader=DataLoder(
        PAWS_X("../../models/data/x-final/ko/translated_train.tsv", "ko_KR",
               "ko_KR"),
        batch_size=32,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
    ),
    val_dataloader=DataLoader(
        PAWS_X("../../models/data/x-final/ko/dev_2k.tsv", "ko_KR", "ko_KR"),
        batch_size=32,
        pin_memory=True,
        num_workers=32,
    ),
)
