import torch
from torch.utils.data import IterableDataset
import pandas as pd


class SequenceDataset(IterableDataset):
    def __init__(self, csv_path, tokenizer, text_column='X', chunksize=1024):
        
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.chunksize = chunksize

    def prep_batch(self,sequence, max_len=256):

        tokens = self.tokenizer.tokenize(sequence)
        token_ids = [self.tokenizer.vocab.get(tok, self.tokenizer.vocab["<unk>"]) for tok in tokens]
        token_ids = token_ids[:max_len]

        pad_len = max_len - len(token_ids)
        if pad_len > 0:
            token_ids += [self.tokenizer.vocab["<pad>"]] * pad_len

        input_ids = torch.tensor([token_ids])  # [1, max_len]
        attention_mask = (input_ids != self.tokenizer.vocab["<pad>"]).long()

        decoder_input_ids_list = [self.tokenizer.vocab["<pad>"]] + token_ids[:-1]
        decoder_input_ids = torch.tensor([decoder_input_ids_list])

        # initialize all column labels with pad tokens
        page_labels       = [self.tokenizer.vocab["<pad>"]] * max_len
        event_labels      = [self.tokenizer.vocab["<pad>"]] * max_len
        category_labels   = [self.tokenizer.vocab["<pad>"]] * max_len
        subcategory_labels= [self.tokenizer.vocab["<pad>"]] * max_len

        for i, tok_id in enumerate(token_ids):
            if tok_id == self.tokenizer.vocab["<pad>"]:
                # nothing to do, already pad
                continue
            pos = i % 4
            if pos == 0:
                page_labels[i] = tok_id
            elif pos == 1:
                event_labels[i] = tok_id
            elif pos == 2:
                category_labels[i] = tok_id
            elif pos == 3:
                subcategory_labels[i] = tok_id

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "page_labels": torch.tensor([page_labels]),
            "event_labels": torch.tensor([event_labels]),
            "category_labels": torch.tensor([category_labels]),
            "subcategory_labels": torch.tensor([subcategory_labels]),
        }
            

    def __iter__(self):
        # Read in chunks
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            for row in chunk.itertuples(index=False):
                sequence = getattr(row, self.text_column)
                if isinstance(sequence, str):
                    batch = self.prep_batch(sequence, self.tokenizer)

                    # Yield each sample
                    yield {
                        "input_ids": batch["input_ids"].squeeze(0),
                        "decoder_input_ids": batch["decoder_input_ids"].squeeze(0),
                        "attention_mask": batch["attention_mask"].squeeze(0),
                        "page_labels": batch["page_labels"].squeeze(0),
                        "event_labels": batch["event_labels"].squeeze(0),
                        "category_labels": batch["category_labels"].squeeze(0),
                        "subcategory_labels": batch["subcategory_labels"].squeeze(0)
                    }
