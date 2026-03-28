import torch
from pathlib import Path
import sentencepiece as spm
from model import build_transformer
from config import get_config, latest_weights_file_path
from dataset import causal_mask


def load_tokenizer(config, lang):
    model_path = config['tokenizer_file'].format(lang) + '.model'
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {model_path}\n"
            f"Please run train.py first to build the tokenizers."
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def load_model(config, tokenizer_src, tokenizer_tgt, device):
    model = build_transformer(
        tokenizer_src.get_piece_size(),
        tokenizer_tgt.get_piece_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise FileNotFoundError(
            "No saved model weights found.\n"
            "Please train the model first using train.py."
        )

    print(f"Loading model weights from: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def translate(sentence: str, model, tokenizer_src, tokenizer_tgt, config, device) -> str:
    """
    Translate a single English sentence to Hindi.
    """
    seq_len = config['seq_len']

    # Tokenize source sentence
    src_tokens = tokenizer_src.encode(sentence)   # List[int]

    # Truncate if longer than seq_len - 2 (for SOS + EOS)
    src_tokens = src_tokens[:seq_len - 2]

    pad_id = tokenizer_src.pad_id()   # 0
    num_padding = seq_len - len(src_tokens) - 2

    # Build encoder input:  [SOS] + tokens + [EOS] + <PAD...>
    encoder_input = torch.cat([
        torch.tensor([tokenizer_src.bos_id()], dtype=torch.int64),
        torch.tensor(src_tokens,               dtype=torch.int64),
        torch.tensor([tokenizer_src.eos_id()], dtype=torch.int64),
        torch.tensor([pad_id] * num_padding,   dtype=torch.int64),
    ], dim=0).unsqueeze(0).to(device)   # (1, seq_len)

    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)  # (1,1,1,seq_len)

    # Greedy decode
    sos_idx = tokenizer_tgt.bos_id()
    eos_idx = tokenizer_tgt.eos_id()

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)

        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

        while True:
            if decoder_input.size(1) == seq_len:
                break

            decoder_mask = (
                causal_mask(decoder_input.size(1))
                .type_as(encoder_mask)
                .to(device)
            )

            out  = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat([
                decoder_input,
                torch.empty(1, 1).fill_(next_word.item()).type_as(encoder_input).to(device)
            ], dim=1)

            if next_word.item() == eos_idx:
                break

    # Decode output tokens to Hindi string
    output_ids = decoder_input.squeeze(0).tolist()
    # Remove SOS token at position 0 before decoding
    translation = tokenizer_tgt.decode(output_ids[1:])
    return translation


def main():
    # ── Setup ────────────────────────────────
    config = get_config()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # ── Load tokenizers & model ───────────────
    print("Loading tokenizers ...")
    tokenizer_src = load_tokenizer(config, config['lang_src'])
    tokenizer_tgt = load_tokenizer(config, config['lang_tgt'])

    print("Loading model ...")
    model = load_model(config, tokenizer_src, tokenizer_tgt, device)

    print("\n" + "="*50)
    print("  English → Hindi Translator")
    print("  Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")

    # ── Interactive loop ──────────────────────
    while True:
        try:
            sentence = input("Enter English sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not sentence:
            continue

        if sentence.lower() in ('quit', 'exit'):
            print("Goodbye!")
            break

        translation = translate(
            sentence, model,
            tokenizer_src, tokenizer_tgt,
            config, device
        )
        print(f"Hindi translation  : {translation}\n")


if __name__ == '__main__':
    main()