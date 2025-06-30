import torch

class Tokenizer:
    def __init__(self, df, columns, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        self.special_tokens = special_tokens

        # Build vocab and masks
        self.vocab, self.idx2token, self.column_masks = self.vocabs(df, columns)
        self.vocab_size = len(self.vocab)
        self.columns = columns

    def vocabs(self, df, columns):
        vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        idx = len(vocab)
        column_tokens = {col: set() for col in columns}

        for col in columns:
            for sequence in df[col]:
                tokens = []
                if isinstance(sequence, str):
                    cleaned = sequence.replace('[', '').replace(']', '').replace("'", '').replace(",", '')
                    tokens = cleaned.strip().split()
                elif isinstance(sequence, list):
                    tokens = sequence
                # else ignore non-str/list
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = idx
                        idx += 1
                    column_tokens[col].add(token)

        idx2token = {v: k for k, v in vocab.items()}

        vocab_size = len(vocab)
        column_masks = {}
        for col in columns:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            for tok in column_tokens[col]:
                mask[vocab[tok]] = True
            # Include special tokens in mask
            for sp in self.special_tokens:
                mask[vocab[sp]] = True
            column_masks[col] = mask

        return vocab, idx2token, column_masks
    def get_vocab_size(self):
        return len(self.vocab)

    def tokenize(self, sequence):
        if isinstance(sequence, str):
            cleaned = sequence.replace('[', '').replace(']', '').replace("'", '').replace(",", '')
            return cleaned.strip().split()
        elif isinstance(sequence, list):
            return [str(x).replace('[', '').replace(']', '').replace("'", '').replace(",", '') for x in sequence]
        else:
            return []

    def encode(self, sequence):
        """
        Convert a sequence of tokens or a string into list of indices using vocab.
        """
        tokens = self.tokenize(sequence)
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

    def decode(self, indices):
        return [self.idx2token.get(idx, "<unk>") for idx in indices]

    def mask(self, column):
        """
        Get the vocab mask tensor for a given column.

        Returns:
            torch.BoolTensor of size vocab_size
        """
        return self.column_masks.get(column, torch.zeros(self.vocab_size, dtype=torch.bool))

