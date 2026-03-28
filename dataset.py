import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len       = seq_len
        self.ds            = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang      = src_lang
        self.tgt_lang      = tgt_lang

        # SentencePiece special token ids (set during SP training)
        # pad=0, unk=1, bos([SOS])=2, eos([EOS])=3
        self.sos_token = torch.tensor([tokenizer_tgt.bos_id()], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.eos_id()], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.pad_id()], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # SentencePiece .encode() returns a plain Python list of ints directly
        # (no .ids needed — that was the HuggingFace tokenizer API)
        enc_input_tokens = self.tokenizer_src.encode(src_text)   # List[int]
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)   # List[int]

        # Padding counts  (we add SOS + EOS to encoder, SOS-only to decoder)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(
                f"Sentence is too long for seq_len={self.seq_len}. "
                f"enc tokens={len(enc_input_tokens)}, "
                f"dec tokens={len(dec_input_tokens)}"
            )

        pad_id = self.pad_token.item()

        # Encoder input:  [SOS] + tokens + [EOS] + <PAD...>
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([pad_id] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Decoder input:  [SOS] + tokens + <PAD...>
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([pad_id] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Label:          tokens + [EOS] + <PAD...>
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([pad_id] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        assert encoder_input.size(0) == self.seq_len, "encoder_input length mismatch"
        assert decoder_input.size(0) == self.seq_len, "decoder_input length mismatch"
        assert label.size(0)         == self.seq_len, "label length mismatch"

        return {
            "encoder_input": encoder_input,   # (seq_len,)
            "decoder_input": decoder_input,   # (seq_len,)
            # (1, 1, seq_len) — True where NOT pad
            "encoder_mask": (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int(),
            # (1, seq_len, seq_len) — pad mask AND causal mask combined
            "decoder_mask": (decoder_input != pad_id).unsqueeze(0).int()
                            & causal_mask(decoder_input.size(0)),
            "label":    label,      # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    """Upper-triangular mask: returns True where attention IS allowed."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0