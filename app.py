"""
app.py — REM Sleep Selectivity Profiler
Streamlit interface: SMILES -> predicted pChEMBL across 6 CNS targets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from predict import load_molformer, load_mlps, validate_smiles, predict, TARGETS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REM Sleep Selectivity Profiler",
    page_icon="🧬",
    layout="wide",
)

# ── Model loading (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading MoLFormer-XL and MLP models (first run only)...")
def get_models():
    tokenizer, encoder = load_molformer()
    mlps = load_mlps()
    return tokenizer, encoder, mlps

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧬 REM Sleep Selectivity Profiler")
st.markdown(
    "Predict binding affinity (pChEMBL) for small molecules across **6 CNS targets** "
    "relevant to REM sleep regulation: **ADORA1 · CHRM2 · CHRM4 · GABRA1 · HCRTR1 · HCRTR2**"
)
st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Input")
    input_method = st.radio(
        "Input method",
        ["Paste SMILES", "Upload .txt file"],
        horizontal=True,
    )

    smiles_raw = []

    if input_method == "Paste SMILES":
        text = st.text_area(
            "Enter one SMILES per line",
            placeholder=(
                "C1CN(CCN1CC2=CC=C(C=C2)C(=O)NC3=CC=C(C=C3)F)C(=O)C4=CC=CS4\n"
                "CN(C)CCOC(c1ccccc1)c1ccccc1"
            ),
            height=160,
        )
        if text.strip():
            smiles_raw = [s.strip() for s in text.strip().splitlines() if s.strip()]
    else:
        uploaded = st.file_uploader("Upload a .txt file (one SMILES per line)", type=["txt"])
        if uploaded:
            smiles_raw = [s.strip() for s in uploaded.read().decode().splitlines() if s.strip()]

    if smiles_raw:
        st.caption(f"{len(smiles_raw)} SMILES detected")

    run = st.button("Predict", type="primary", disabled=len(smiles_raw) == 0)

with col_right:
    st.subheader("Example compounds")
    examples = {
        "Suvorexant (HCRTR1/2 antagonist)":
            "C[C@@H]1CCN(CCN1C(=O)C2=C(C=CC(=C2)C)N3N=CC=N3)C4=NC5=C(O4)C=CC(=C5)Cl",
        "Scopolamine (mAChR antagonist)":
            "CN1[C@@H]2CC(C[C@H]1[C@H]3[C@@H]2O3)OC(=O)[C@H](CO)C4=CC=CC=C4",
        "Diazepam (GABRA1 modulator)":
            "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
        "DPCPX (ADORA1 antagonist)":
            "CCCN1C2=C(C(=O)N(C1=O)CCC)NC(=N2)C3CCCC3",
    }
    for label, smi in examples.items():
        st.code(smi, language=None)
        st.caption(label)

st.divider()

# ── Prediction + Results ───────────────────────────────────────────────────────
if run and smiles_raw:
    canonical, invalid = validate_smiles(smiles_raw)

    if invalid:
        st.warning(
            f"Skipped {len(invalid)} invalid SMILES: "
            + ", ".join(f"`{s}`" for s in invalid[:5])
            + ("..." if len(invalid) > 5 else "")
        )

    if not canonical:
        st.error("No valid SMILES to process.")
        st.stop()

    with st.spinner(f"Generating predictions for {len(canonical)} molecule(s)..."):
        tokenizer, encoder, mlps = get_models()
        results = predict(canonical, tokenizer, encoder, mlps)

    # Build dataframe
    df = pd.DataFrame(
        {t: results[t] for t in TARGETS},
        index=[smi[:60] + "..." if len(smi) > 60 else smi for smi in canonical],
    )
    df.index.name = "SMILES"
    df = df.round(3)

    # ── Table ──────────────────────────────────────────────────────────────────
    st.subheader("Predicted pChEMBL Values")
    st.markdown("Color scale: red (low) → yellow → green (high). Values are pChEMBL — higher = stronger predicted binding.")

    styled = df.style.background_gradient(cmap="RdYlGn", axis=None, vmin=4, vmax=8).format("{:.3f}")
    st.dataframe(styled, use_container_width=True)

    csv = df.to_csv()
    st.download_button(
        "⬇ Download as CSV",
        data=csv,
        file_name="selectivity_predictions.csv",
        mime="text/csv",
    )

    # ── Chart ──────────────────────────────────────────────────────────────────
    st.subheader("Selectivity Profile")

    COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    if len(canonical) == 1:
        vals = [float(results[t][0]) for t in TARGETS]
        max_val = max(vals)
        bar_colors = ["#2ecc71" if v == max_val else "#3498db" for v in vals]
        fig = go.Figure(
            go.Bar(x=TARGETS, y=vals, marker_color=bar_colors, text=[f"{v:.3f}" for v in vals], textposition="outside")
        )
        smi_label = canonical[0][:50] + "..." if len(canonical[0]) > 50 else canonical[0]
        fig.update_layout(
            title=f"Selectivity profile: {smi_label}",
            xaxis_title="Target",
            yaxis_title="Predicted pChEMBL",
            yaxis=dict(range=[0, 9]),
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
    else:
        fig = go.Figure()
        for i, smi in enumerate(canonical):
            label = smi[:35] + "..." if len(smi) > 35 else smi
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=TARGETS,
                    y=[float(results[t][i]) for t in TARGETS],
                    text=[f"{float(results[t][i]):.2f}" for t in TARGETS],
                    textposition="outside",
                )
            )
        fig.update_layout(
            barmode="group",
            xaxis_title="Target",
            yaxis_title="Predicted pChEMBL",
            yaxis=dict(range=[0, 9]),
            legend_title="Molecule",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig, use_container_width=True)

# ── Caveats ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Model reliability and caveats", expanded=False):
    st.markdown(
        """
**Model performance (Spearman R, held-out test set, 5-seed average):**

| Target | Spearman R | RMSE | Notes |
|--------|-----------|------|-------|
| HCRTR1 | 0.68 ± 0.02 | 0.67 | Most reliable |
| HCRTR2 | 0.63 ± 0.01 | 0.75 | Reliable |
| CHRM2  | 0.61 ± 0.08 | 1.47 | Moderate |
| GABRA1 | 0.61 ± 0.09 | 0.88 | Moderate |
| ADORA1 | 0.38 ± 0.01 | 0.96 | Limited reliability |
| CHRM4  | 0.19 ± 0.06 | 1.06 | Unreliable — treat as exploratory |

**Important caveats:**
- Predictions are for **research purposes only**, not clinical or regulatory use.
- Models are MoLFormer-XL (IBM) + MLP regressors trained on ChEMBL bioactivity data.
- Training used a Butina cluster split (Tanimoto threshold 0.6) to prevent data leakage.
- pChEMBL is a log-scale: a difference of 1 unit = 10x affinity difference.
- Predictions for compounds far outside the training distribution (novel scaffolds) may be unreliable.
- CHRM4 (R=0.19) predictions carry high uncertainty and should be interpreted cautiously.
- Some cross-target contamination exists — e.g., the CHRM2 model partly reflects CHRM4 activity.
        """
    )

st.caption("Built with MoLFormer-XL (IBM Research) · MLP regressors trained on ChEMBL data · UCD Research Project 2026")
