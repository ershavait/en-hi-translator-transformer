from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
import os
import random
from pathlib import Path

from datasets import load_dataset
import sentencepiece as spm
import torchmetrics


# ─────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────

def get_or_build_tokenizer(config, ds, lang):
    """
    Train a SentencePiece BPE tokenizer if one doesn't exist, then load it.
    character_coverage is automatically set to 1.0 for Hindi (Devanagari)
    and 0.9995 for English (Latin).
    """
    model_prefix = config['tokenizer_file'].format(lang)
    model_path   = model_prefix + '.model'

    if not Path(model_path).exists():
        tmp_file = f'/tmp/sp_train_{lang}.txt'
        print(f"[Tokenizer] Writing '{lang}' sentences to {tmp_file} ...")
        with open(tmp_file, 'w', encoding='utf-8') as f:
            for item in ds:
                line = item['translation'][lang].strip()
                if line:
                    f.write(line + '\n')

        coverage = 1.0 if lang == config['lang_tgt'] else config.get('character_coverage', 0.9995)

        print(f"[Tokenizer] Training SentencePiece for '{lang}' (coverage={coverage}) ...")
        spm.SentencePieceTrainer.train(
            input=tmp_file,
            model_prefix=model_prefix,
            vocab_size=config.get('vocab_size', 16000),
            character_coverage=coverage,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[SOS]',
            eos_piece='[EOS]',
        )
        print(f"[Tokenizer] Saved → {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


# ─────────────────────────────────────────────
#  Filter & subsample dataset
# ─────────────────────────────────────────────

def filter_and_sample(ds, tokenizer_src, tokenizer_tgt,
                      src_lang, tgt_lang,
                      max_len, sample_percent):
    """
    1. Filter out sentences longer than max_len tokens
    2. Randomly keep only sample_percent % of what's left
    """
    filtered = []
    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][src_lang])
        tgt_ids = tokenizer_tgt.encode(item['translation'][tgt_lang])
        if len(src_ids) <= max_len - 2 and len(tgt_ids) <= max_len - 1:
            filtered.append(item)

    # Randomly sample the requested percentage
    sample_size = int(len(filtered) * sample_percent)
    sampled = random.sample(filtered, sample_size)

    return sampled


# ─────────────────────────────────────────────
#  Greedy decoder
# ─────────────────────────────────────────────

def greedy_decode(model, source, source_mask,
                  tokenizer_src, tokenizer_tgt,
                  max_len, device):
    sos_idx = tokenizer_tgt.bos_id()
    eos_idx = tokenizer_tgt.eos_id()

    encoder_output = model.encode(source, source_mask)

    decoder_input = (
        torch.empty(1, 1)
              .fill_(sos_idx)
              .type_as(source)
              .to(device)
    )

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = (
            causal_mask(decoder_input.size(1))
            .type_as(source_mask)
            .to(device)
        )

        out  = model.decode(encoder_output, source_mask,
                            decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1)
                  .type_as(source)
                  .fill_(next_word.item())
                  .to(device)
        ], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────

def run_validation(model, validation_ds,
                   tokenizer_src, tokenizer_tgt,
                   max_len, device,
                   print_msg, global_step, writer,
                   num_examples=2):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask  = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, \
                "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt,
                max_len, device
            )

            source_text    = batch['src_text'][0]
            target_text    = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().tolist()
            )

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

def get_ds(config):
    # IITB only has 'default' config — do NOT pass 'en-hi'
    print("[Data] Loading IITB train split ...")
    ds_train_raw = load_dataset(config['datasource'], split='train')

    print("[Data] Loading IITB validation split ...")
    ds_val_raw   = load_dataset(config['datasource'], split='validation')

    # Build tokenizers from full training data
    tokenizer_src = get_or_build_tokenizer(config, ds_train_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_train_raw, config['lang_tgt'])

    # Filter long sentences + keep only 8% of data
    sample_pct = config.get('sample_percent', 0.0)
    max_len    = config['seq_len']

    print(f"[Data] Filtering long sentences (max_len={max_len}) and sampling {sample_pct*100:.0f}% ...")
    ds_train_sampled = filter_and_sample(
        ds_train_raw, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'],
        max_len, sample_pct
    )
    ds_val_sampled = filter_and_sample(
        ds_val_raw, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'],
        max_len, 1.0   # keep all validation sentences that fit
    )

    print(f"  Train size after filter+sample : {len(ds_train_sampled):,}")
    print(f"  Val size after filter          : {len(ds_val_sampled):,}")

    train_ds = BilingualDataset(
        ds_train_sampled, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )
    val_ds = BilingualDataset(
        ds_val_sampled, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len,
        config['seq_len'], config['seq_len'],
        d_model=config['d_model']
    )
    return model


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────

def train_model(config):
    # ── Device ───────────────────────────────
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch, 'has_mps', False) or torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    if device == 'cuda':
        idx = torch.device(device).index or 0
        print(f"  Name  : {torch.cuda.get_device_name(idx)}")
        print(f"  Memory: {torch.cuda.get_device_properties(idx).total_memory / 1024**3:.1f} GB")
    elif device == 'mps':
        print("  Name  : <mps>")
    else:
        print("  NOTE  : No GPU — training will be slow on CPU.")

    device = torch.device(device)

    # ── Folders ───────────────────────────────
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    # ── Data ─────────────────────────────────
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # ── Model ────────────────────────────────
    model = get_model(
        config,
        tokenizer_src.get_piece_size(),
        tokenizer_tgt.get_piece_size()
    ).to(device)

    writer    = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # ── Checkpoint preload ───────────────────
    initial_epoch = 0
    global_step   = 0
    preload       = config['preload']

    if preload == 'latest':
        model_filename = latest_weights_file_path(config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    if model_filename:
        print(f"[Checkpoint] Preloading → {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("[Checkpoint] No preloaded model — starting from scratch.")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.pad_id(),
        label_smoothing=0.1
    ).to(device)

    # ── Epoch loop ───────────────────────────
    for epoch in range(initial_epoch, config['num_epochs']):
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask  = batch['encoder_mask'].to(device)
            decoder_mask  = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask,
                decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_piece_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({'loss': f'{loss.item():.3f}'})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # ── Validation ───────────────────────
        run_validation(
            model, val_dataloader,
            tokenizer_src, tokenizer_tgt,
            config['seq_len'], device,
            lambda msg: batch_iterator.write(msg),
            global_step, writer
        )

        # ── Save checkpoint ──────────────────
        ckpt_path = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step':          global_step,
        }, ckpt_path)
        print(f"[Checkpoint] Saved → {ckpt_path}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)