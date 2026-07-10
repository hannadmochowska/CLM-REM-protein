"""
Cross-target specificity matrix for 6 REM sleep-regulatory targets.
Fits each locked model on its own training set, predicts on all 6 test sets.
Uses a GLOBAL max_dissimilarity_2 split (all 6 compound sets pooled, then test
set built by MaxMin diversity picking) to prevent cross-dataset compound leakage
and achieve balanced ~20% test fractions per target.
Output: 6x6 Pearson R matrix saved as specificity_matrix.csv
"""

import sqlite3
import numpy as np
import pandas as pd
import torch, types, math
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = "chembl_36/chembl_36_sqlite/chembl_36.db"

TARGETS = {
    "ADORA1":  {"chembl_id": "CHEMBL226",  "arch": (512,256,128), "act": "tanh", "alpha": 0.01,   "lr": 0.001},
    "CHRM2":   {"chembl_id": "CHEMBL211",  "arch": (512,256,128), "act": "relu", "alpha": 0.01,   "lr": 0.0001},
    "CHRM4":   {"chembl_id": "CHEMBL1821", "arch": (512,256,128), "act": "relu", "alpha": 0.01,   "lr": 0.0001},
    "GABRA1":  {"chembl_id": "CHEMBL1962", "arch": (512,256,128), "act": "relu", "alpha": 0.001,  "lr": 0.001},
    "HCRTR1":  {"chembl_id": "CHEMBL5113", "arch": (512,256,128), "act": "tanh", "alpha": 0.001,  "lr": 0.001},
    "HCRTR2":  {"chembl_id": "CHEMBL4792", "arch": (512,256,128), "act": "tanh", "alpha": 0.0001, "lr": 0.0005},
}

SEEDS = [42, 7, 13, 99, 21]
TARGET_NAMES = list(TARGETS.keys())

# ── Step 1: Load and clean all 6 datasets ─────────────────────────────────────

print("=" * 60)
print("Step 1: Loading datasets from ChEMBL")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

query_template = """
SELECT cs.canonical_smiles, act.pchembl_value
FROM activities act
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE td.chembl_id = '{chembl_id}'
  AND act.pchembl_value IS NOT NULL
  AND act.standard_relation = '='
"""

datasets = {}
for name, cfg in TARGETS.items():
    df = pd.read_sql_query(query_template.format(chembl_id=cfg["chembl_id"]), conn)
    df = df.groupby("canonical_smiles")["pchembl_value"].median().reset_index()
    df = df[df["pchembl_value"] >= 5].reset_index(drop=True)
    datasets[name] = df
    print(f"  {name}: {len(df)} compounds")

conn.close()

# ── Step 2: Load MoLFormer-XL once ────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2: Loading MoLFormer-XL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
molformer = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    deterministic_eval=True, trust_remote_code=True)
molformer.eval()
molformer.get_head_mask = lambda head_mask, num_layers, is_attention_chunked=False: [None] * num_layers

# Patch attention (PR#7 fix)
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
        mask = (attention_mask == 0).to(attention_mask.dtype)
        per_pos = mask[:, 0, -1]
        k = k * per_pos[:, None, -L:, None]
    kv   = torch.matmul(k.transpose(-1, -2), v)
    norm = torch.matmul(q, k.sum(dim=-2).unsqueeze(-1)).clamp(min=self.eps)
    ctx  = torch.matmul(q, kv) / norm
    ctx  = ctx.permute(0, 2, 1, 3).contiguous()
    ctx  = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
    return (ctx,)

for layer in molformer.encoder.layer:
    layer.attention.self.forward = types.MethodType(_fixed_attn_fwd, layer.attention.self)
print("Patched all 12 attention layers")

def get_embeddings_batched(smiles_list, batch_size=64):
    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = molformer(**inputs)
        all_embs.append(outputs.pooler_output.cpu().numpy())
    return np.vstack(all_embs)

# ── Step 3: Generate embeddings for all 6 datasets ────────────────────────────

print("\n" + "=" * 60)
print("Step 3: Generating embeddings")
print("=" * 60)

embeddings = {}
for name, df in datasets.items():
    print(f"  {name} ({len(df)} compounds)...")
    emb = get_embeddings_batched(df["canonical_smiles"].tolist())
    embeddings[name] = emb
    print(f"    Shape: {emb.shape}  NaN: {np.isnan(emb).any()}")

# ── Step 4: Global max_dissimilarity_2 split ──────────────────────────────────

print("\n" + "=" * 60)
print("Step 4: Global max_dissimilarity_2 split (pooled across all 6 targets)")
print("=" * 60)

# Pool all unique SMILES across all targets
all_smiles = sorted(set(smi for df in datasets.values() for smi in df["canonical_smiles"]))
n = len(all_smiles)
print(f"  Total unique compounds across all targets: {n}")

# Compute ECFP4 fingerprints for pooled set
fps = []
for smi in all_smiles:
    mol = Chem.MolFromSmiles(smi)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

# Build full pairwise Tanimoto similarity matrix
# (~1.2 GB for n~17k; takes ~30 seconds)
print("  Computing pairwise Tanimoto similarity matrix...")
sim = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    sim[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
print(f"  Similarity matrix built: {sim.shape}")

# max_dissimilarity_2: build test set by MaxMin diversity picking.
# Iteratively selects the compound most similar to the current test set,
# ensuring the test set spans the extremes of chemical space.
# No threshold applied (threshold=None as per Raul's suggestion).
n_test_target = int(0.2 * n)

# Seed the test set with the compound least similar to all others on average
ood_seed = int(np.argmin(sim.mean(axis=1)))
in_test = np.zeros(n, dtype=bool)
in_test[ood_seed] = True
max_sim_to_test = sim[ood_seed].copy()  # running max Tanimoto to any test compound

print(f"  Selecting {n_test_target} test compounds using MaxMin diversity picking...")
for step in range(n_test_target - 1):
    candidates = np.where(~in_test)[0]
    best = int(candidates[max_sim_to_test[candidates].argmax()])
    in_test[best] = True
    max_sim_to_test = np.maximum(max_sim_to_test, sim[best])
    if step % 500 == 0:
        print(f"    {step}/{n_test_target} compounds selected...")

global_test_smiles = set(all_smiles[i] for i in range(n) if in_test[i])
print(f"  Global test set: {in_test.sum()} / {n} compounds ({in_test.mean()*100:.1f}%)")

# Apply global split to each target
splits = {}
for name, df in datasets.items():
    smiles = df["canonical_smiles"].tolist()
    train_idx = np.array([i for i, s in enumerate(smiles) if s not in global_test_smiles])
    test_idx  = np.array([i for i, s in enumerate(smiles) if s in global_test_smiles])
    splits[name] = (train_idx, test_idx)
    print(f"  {name}: train={len(train_idx)}  test={len(test_idx)}  ({100*len(test_idx)/len(smiles):.1f}% test)")

# Verify zero cross-dataset leakage
print("\n  Cross-dataset leakage check (all should be 0):")
for t1, t2 in combinations(TARGET_NAMES, 2):
    train1 = set(datasets[t1]["canonical_smiles"].iloc[splits[t1][0]])
    test2  = set(datasets[t2]["canonical_smiles"].iloc[splits[t2][1]])
    train2 = set(datasets[t2]["canonical_smiles"].iloc[splits[t2][0]])
    test1  = set(datasets[t1]["canonical_smiles"].iloc[splits[t1][1]])
    overlap_a = len(train1 & test2)
    overlap_b = len(train2 & test1)
    status = "OK" if overlap_a == 0 and overlap_b == 0 else "LEAKAGE DETECTED"
    print(f"    {t1}/{t2}: {overlap_a}, {overlap_b}  [{status}]")

# ── Step 5: Build 6x6 specificity matrix ──────────────────────────────────────

print("\n" + "=" * 60)
print("Step 5: Building 6x6 specificity matrix")
print("=" * 60)

matrix = np.zeros((6, 6))

for i, model_target in enumerate(TARGET_NAMES):
    cfg = TARGETS[model_target]
    train_idx, _ = splits[model_target]
    X_train = embeddings[model_target][train_idx]
    y_train = datasets[model_target]["pchembl_value"].values[train_idx]

    print(f"\n  Model: {model_target} (train n={len(train_idx)})")

    for j, test_target in enumerate(TARGET_NAMES):
        _, test_idx = splits[test_target]
        X_test = embeddings[test_target][test_idx]
        y_test = datasets[test_target]["pchembl_value"].values[test_idx]

        rs = []
        for seed in SEEDS:
            mlp = MLPRegressor(
                hidden_layer_sizes=cfg["arch"],
                activation=cfg["act"],
                alpha=cfg["alpha"],
                learning_rate_init=cfg["lr"],
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed
            )
            mlp.fit(X_train, y_train)
            r, _ = pearsonr(mlp.predict(X_test), y_test)
            rs.append(r)

        mean_r = np.mean(rs)
        matrix[i, j] = mean_r
        marker = " <-- diagonal" if i == j else ""
        print(f"    vs {test_target}: R={mean_r:.4f}  Std={np.std(rs):.4f}{marker}")

# ── Step 6: Save and print results ────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 6: Results")
print("=" * 60)

df_matrix = pd.DataFrame(matrix, index=TARGET_NAMES, columns=TARGET_NAMES)
df_matrix.index.name = "Model \\ Test"

print("\n6x6 Pearson R Specificity Matrix")
print("(rows = model trained on, columns = test set predicted on)\n")
print(df_matrix.round(4).to_string())

df_matrix.to_csv("specificity_matrix.csv")
print("\nSaved to specificity_matrix.csv")
