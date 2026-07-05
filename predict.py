"""
predict.py — REM Sleep Selectivity Profiler
Standalone inference pipeline: SMILES -> MoLFormer-XL embedding -> 6 MLP predictions

Usage:
    python predict.py "CC(=O)Oc1ccccc1C(=O)O"
    python predict.py smiles.txt        (one SMILES per line)
"""

import sys
import types
import math
import numpy as np
import joblib
import torch
from typing import Optional
from pathlib import Path

# ── RDKit for validation ───────────────────────────────────────────────────────
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: rdkit not installed. SMILES validation disabled.")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID   = "ibm-research/MoLFormer-XL-both-10pct"
MODEL_REV  = "e2ac555fb16948735e52c76111c26b6fb8873856"  # last revision compatible with transformers<4.44
TARGETS    = ["ADORA1", "CHRM2", "CHRM4", "GABRA1", "HCRTR1", "HCRTR2"]
MODEL_DIR  = Path(__file__).parent   # expects mlp_*.joblib in same folder as this script


# ── MoLFormer loader (with PR#7 attention patch, matching your notebooks) ──────
def load_molformer():
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True)
    encoder   = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REV, deterministic_eval=True, trust_remote_code=True
    )
    encoder.eval()
    encoder.get_head_mask = lambda head_mask, num_layers, is_attention_chunked=False: [None] * num_layers

    # PR#7 rotary + linear attention fix (identical to your notebook)
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(q, k, cos, sin, pos_ids):
        cos = cos[pos_ids].unsqueeze(1)
        sin = sin[pos_ids].unsqueeze(1)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    def _fixed_attn_fwd(self, hidden_states, attention_mask=None, position_ids=None,
                        head_mask=None, output_attentions=False):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        L = k.shape[-2]
        cos, sin = self.rotary_embeddings(v, seq_len=L)
        q, k = _apply_rotary(q, k, cos, sin, position_ids)
        q, k = self.feature_map(q, k)
        if attention_mask is not None:
            mask    = (attention_mask == 0).to(attention_mask.dtype)
            per_pos = mask[:, 0, -1]
            k       = k * per_pos[:, None, -L:, None]
        kv   = torch.matmul(k.transpose(-1, -2), v)
        norm = torch.matmul(q, k.sum(dim=-2).unsqueeze(-1)).clamp(min=self.eps)
        ctx  = torch.matmul(q, kv) / norm
        ctx  = ctx.permute(0, 2, 1, 3).contiguous()
        ctx  = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        return (ctx,)

    for layer in encoder.encoder.layer:
        layer.attention.self.forward = types.MethodType(_fixed_attn_fwd, layer.attention.self)

    print(f"MoLFormer-XL loaded and patched ({sum(p.numel() for p in encoder.parameters()):,} params)")
    return tokenizer, encoder


# ── SMILES validation ──────────────────────────────────────────────────────────
def validate_smiles(smiles_list):
    """Returns (canonical_list, invalid_list). Passes through if RDKit unavailable."""
    if not RDKIT_AVAILABLE:
        return smiles_list, []
    valid, invalid, canonical = [], [], []
    for smi in smiles_list:
        smi = smi.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            valid.append(smi)
            canonical.append(Chem.MolToSmiles(mol))
    return canonical, invalid


# ── Embedding generation ───────────────────────────────────────────────────────
def embed(smiles_list, tokenizer, encoder, batch_size=1):
    """Returns np.ndarray of shape (n, 768) using pooler_output — matches training."""
    all_embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = encoder(**inputs)
        all_embeddings.append(outputs.pooler_output.cpu().numpy())
    return np.vstack(all_embeddings)


# ── MLP loader ─────────────────────────────────────────────────────────────────
def load_mlps(model_dir=MODEL_DIR):
    mlps = {}
    for t in TARGETS:
        path = model_dir / f"mlp_{t}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        mlps[t] = joblib.load(path)
    print(f"Loaded {len(mlps)} MLP models: {list(mlps.keys())}")
    return mlps


# ── Main prediction function ───────────────────────────────────────────────────
def predict(smiles_list, tokenizer, encoder, mlps):
    """
    Args:
        smiles_list: list of SMILES strings (already validated/canonicalised)
        tokenizer, encoder: MoLFormer-XL
        mlps: dict {target: MLPRegressor}
    Returns:
        dict {target: np.ndarray of predicted pChEMBL values}
    """
    embeddings = embed(smiles_list, tokenizer, encoder)
    return {t: mlps[t].predict(embeddings) for t in TARGETS}


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Parse input
    if len(sys.argv) < 2:
        # Default sanity-check compounds
        smiles_input = [
            "C[C@@H]1CCN(CCN1C(=O)C2=C(C=CC(=C2)C)N3N=CC=N3)C4=NC5=C(O4)C=CC(=C5)Cl",  # suvorexant (HCRTR1/2) — PubChem CID 24965990
            "CN1[C@@H]2CC(C[C@H]1[C@H]3[C@@H]2O3)OC(=O)[C@H](CO)C4=CC=CC=C4",           # scopolamine (CHRM2/4) — PubChem CID 3000322
            "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",                                  # diazepam (GABRA1) — PubChem CID 3016
            "CCCN1C2=C(C(=O)N(C1=O)CCC)NC(=N2)C3CCCC3",                                   # DPCPX (ADORA1) — PubChem CID 1329
        ]
        labels = ["suvorexant", "scopolamine", "diazepam", "DPCPX"]
    elif sys.argv[1].endswith(".txt"):
        with open(sys.argv[1]) as f:
            smiles_input = [l.strip() for l in f if l.strip()]
        labels = smiles_input
    else:
        smiles_input = [sys.argv[1]]
        labels = smiles_input

    # Validate
    canonical, invalid = validate_smiles(smiles_input)
    if invalid:
        print(f"\nInvalid SMILES (skipped): {invalid}")
    if not canonical:
        print("No valid SMILES to process.")
        sys.exit(1)

    # Load models
    print("\nLoading MoLFormer-XL...")
    tokenizer, encoder = load_molformer()
    mlps = load_mlps()

    # Predict
    print(f"\nGenerating embeddings for {len(canonical)} molecule(s)...")
    results = predict(canonical, tokenizer, encoder, mlps)

    # Display
    print("\n" + "="*70)
    print(f"{'SMILES':<35} " + " ".join(f"{t:>8}" for t in TARGETS))
    print("="*70)
    for i, smi in enumerate(canonical):
        label = smi[:33] + ".." if len(smi) > 35 else smi
        preds = " ".join(f"{results[t][i]:>8.3f}" for t in TARGETS)
        print(f"{label:<35} {preds}")
    print("="*70)
    print("Values are predicted pChEMBL. Higher = stronger predicted binding.")
    print("Reliability: HCRTR1/HCRTR2 (R≈0.68) > CHRM2/GABRA1 (R≈0.61) > ADORA1 (R=0.38) > CHRM4 (R=0.19)")
