from pathlib import Path


def get_config():
    return {
        # ── Dataset ──────────────────────────────────────────
        "datasource":         "cfilt/iitb-english-hindi",
        "lang_src":           "en",
        "lang_tgt":           "hi",

        # ── Tokenizer ────────────────────────────────────────
        "tokenizer_file":     "tokenizer_{0}",   # → tokenizer_en.model / tokenizer_hi.model
        "vocab_size":         16000,
        # 0.9995 for English (Latin), auto-overridden to 1.0 for Hindi in train.py
        "character_coverage": 0.9995,

        # ── Model architecture ───────────────────────────────
        "d_model":            512,
        "seq_len":            150,       # ← filters out very long sentences, much faster

        # ── Training ─────────────────────────────────────────
        "batch_size":         32,        # ← L4 22GB handles this comfortably
        "sample_percent":     0.05,      # ← use 5% of data (~330K sentences)
        "lr":                 1e-4,
        "num_epochs":         20,

        # ── Checkpoints ──────────────────────────────────────
        "model_folder":       "weights",
        "model_basename":     "tmodel_",
        # 'latest' → auto-resume from last saved epoch
        # None     → always start from scratch
        "preload":            "latest",

        # ── Tensorboard ──────────────────────────────────────
        "experiment_name":    "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder   = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_files  = list(Path(model_folder).glob(f"{config['model_basename']}*.pt"))
    if not model_files:
        return None
    # Sort by filename — tmodel_00.pt < tmodel_01.pt etc.
    model_files.sort()
    return str(model_files[-1])