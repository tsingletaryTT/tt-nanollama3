import sys, os
sys.path.insert(0, "/home/ttuser/code/tt-tnt")
for k,v in (("TT_METAL_HOME","/home/ttuser/tt-metal"),("TT_METAL_RUNTIME_ROOT","/home/ttuser/tt-metal"),
            ("TT_METAL_ARCH_NAME","blackhole"),("TT_LOGGER_LEVEL","FATAL")): os.environ.setdefault(k,v)
sys.path.append("/home/ttuser/tt-metal/tt-train/sources/ttml")
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
from train.config import build_yaml_config
import train.model as tt_tnt_model
from train.enthusiasts import (MoEHyperparams, install_enthusiasts, warm_start,
                               enthusiast_of_token)
from ttml.common.utils import initialize_device, set_seed

cfg = build_yaml_config(tokenizer_dir="artifacts/tokenizer", model_config_path="x.yaml",
                        seq_len=512, max_sequence_length=512, batch_size=2, max_steps=2)
initialize_device(cfg); set_seed(5489)
tc = {"model_type":"llama","num_heads":16,"num_groups":4,"embedding_dim":1024,
      "dropout_prob":0.0,"num_blocks":8,"vocab_size":32000,
      "max_sequence_length":512,"runner_type":"default","theta":500000.0}
model = tt_tnt_model.create_model(cfg, tc)

emb = load_file("artifacts/hf-tt-tnt-1024-dialogue/model.safetensors")["model.embed_tokens.weight"].float().numpy()
per_tok = enthusiast_of_token(balance=True)
hp = MoEHyperparams(dim=1024, moe_inter_dim=928, n_routed_experts=10, n_activated_experts=2)
summary = install_enthusiasts(model, hp, gate_policy="frozen", first_moe_block=2,
                              reference_embedding=emb, per_token_expert=per_tok)
ws = warm_start(model, Path("artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl"), transformer_config=tc, yaml_config=cfg, moe_block_indices=summary['moe_blocks'])
print("\nfresh parameters (should be experts and gates only):")
for n in ws["fresh_examples"]: print("   ", n)
