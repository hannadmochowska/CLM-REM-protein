"""
app.py — REM Sleep Selectivity Profiler
Streamlit interface implementing design handoff "2a" (warm lab report on white).

SMILES -> predicted pChEMBL across 6 CNS targets, presented as:
header + target pills, input/method, example compounds, heatmap table,
interactive grouped-bar selectivity chart, and a collapsible reliability panel.

Design layer follows design_handoff_selectivity_profiler / option 2a exactly.
Prediction stays in Python (predict.py: MoLFormer-XL + MLP regressors); the
heatmap table and chart appear once a prediction succeeds.
"""

import json
import streamlit as st
import streamlit.components.v1 as components

from predict import validate_smiles, TARGETS  # heavy model imports are lazy (see get_models)

# ── Design tokens ───────────────────────────────────────────────────────────────
ACCENT   = "#1f3a5f"                                        # navy (themeable --accent)
CCOLORS  = ["#1f3a5f", "#6f9fc0", "#b5502f", "#d99a80"]     # compound series (Warm palette)
TIER_COL = {"high": "#2f9e63", "mid": "#c98a1e", "low": "#c9485b"}

EXAMPLES = [
    {"name": "Suvorexant", "role": "HCRTR1/2 antagonist",
     "smiles": "C[C@@H]1CCN(CCN1C(=O)C2=C(C=CC(=C2)C)N3N=CC=N3)C4=NC5=C(O4)C=CC(=C5)Cl"},
    {"name": "Scopolamine", "role": "mAChR antagonist",
     "smiles": "CN1[C@@H]2CC(C[C@H]1[C@H]3[C@@H]2O3)OC(=O)[C@H](CO)C4=CC=CC=C4"},
    {"name": "Diazepam", "role": "GABRA1 modulator",
     "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"},
]

RELIAB = [
    {"t": "HCRTR1", "r": "0.820", "rmse": "0.67", "note": "Most reliable", "tier": "high", "w": "82%"},
    {"t": "CHRM2",  "r": "0.609", "rmse": "1.47", "note": "Reliable",      "tier": "high", "w": "61%"},
    {"t": "GABRA1", "r": "0.587", "rmse": "0.88", "note": "Moderate",      "tier": "mid",  "w": "59%"},
    {"t": "CHRM4",  "r": "0.573", "rmse": "1.06", "note": "Moderate",      "tier": "mid",  "w": "57%"},
    {"t": "HCRTR2", "r": "0.513", "rmse": "0.75", "note": "Moderate",      "tier": "mid",  "w": "51%"},
    {"t": "ADORA1", "r": "0.417", "rmse": "0.96", "note": "Limited",       "tier": "low",  "w": "42%"},
]

CAVEATS = [
    "Predictions are for research purposes only, not for clinical or regulatory use.",
    "Models are MoLFormer-XL (IBM) embeddings + MLP regressors trained on ChEMBL bioactivity data.",
    "Performance measured on a global CCPart split (17,149 compounds pooled across all 6 targets before Butina clustering, Tanimoto 0.6) to prevent cross-dataset leakage.",
    "CHRM2 and CHRM4 models share substantial signal: high cross-target generalisation (R=0.562) reflects overlapping pharmacology at muscarinic receptors.",
    "IC50-equivalent values are estimated as 10^(9 − pChEMBL) nM. pChEMBL is a standardised −log10(molar) scale that aggregates IC50, Ki, Kd and EC50 measurements — the displayed IC50 is an approximation and may not match a direct IC50 assay.",
]

# ── Heatmap color math (ported from the prototype Component) ────────────────────
def _heat_rgb(v: float):
    d0, d1 = 4.5, 9.0
    t = max(0.0, min(1.0, (v - d0) / (d1 - d0)))
    stops = [(201, 72, 91), (233, 196, 106), (47, 158, 99)]
    if t < 0.5:
        a, b, tt = stops[0], stops[1], t * 2
    else:
        a, b, tt = stops[1], stops[2], (t - 0.5) * 2
    return tuple(round(a[i] + (b[i] - a[i]) * tt) for i in range(3))

def heat_bg(v: float) -> str:
    return "#%02x%02x%02x" % _heat_rgb(v)

def heat_fg(v: float) -> str:
    r, g, b = _heat_rgb(v)
    return "#2a2620" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"

def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

def pchembl_to_ic50_str(v: float) -> str:
    """Convert pChEMBL to a human-readable IC50-equivalent string."""
    nm = 10 ** (9 - v)
    if nm < 1:
        return f"{nm * 1000:.0f} pM"
    elif nm < 1000:
        return f"{nm:.1f} nM"
    else:
        return f"{nm / 1000:.1f} µM"

# ── Model loading (cached; imports torch/transformers lazily) ───────────────────
@st.cache_resource(show_spinner="Loading MoLFormer-XL and MLP models (first run only)…")
def get_models():
    from predict import load_molformer, load_mlps
    tokenizer, encoder = load_molformer()
    mlps = load_mlps()
    return tokenizer, encoder, mlps

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="REM Sleep Selectivity Profiler", page_icon="🧬", layout="wide")

# ── Global CSS: fonts, tokens, card shell, native-widget styling ────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap');

:root { --accent: #1f3a5f; }

/* Page background behind the card */
[data-testid="stAppViewContainer"], .stApp { background: #eae6dd; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }

/* .block-container == the 1160px white report card */
.block-container {
    max-width: 1160px !important;
    padding: 0 58px 56px !important;
    margin: 34px auto !important;
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,.10);
    overflow-x: clip;   /* keep rounded corners; allow the page to scroll vertically */
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: #2a2620;
}

/* Collapse Streamlit's default vertical rhythm; sections space themselves */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stElementContainer"] { margin: 0 !important; }
div[data-testid="stMarkdownContainer"] p { margin: 0; }

/* iframes (components.html) sit flush inside the card padding */
iframe { border: none !important; display: block; }
[data-testid="stIFrame"] { line-height: 0; }

/* ── Input textarea ── */
.stTextArea textarea {
    background: #faf9f6 !important;
    border: 1px solid #e2ddd3 !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    line-height: 1.85 !important;
    color: #443d30 !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; }
.stTextArea textarea::placeholder { color: #b3ab9a !important; }
.stTextArea label, .stRadio label[data-testid="stWidgetLabel"] { display: none !important; }
.stTextArea div[data-baseweb="textarea"] { border: none !important; background: transparent !important; }

/* ── Method radios ── */
[data-testid="stRadio"] div[role="radiogroup"] { gap: 10px !important; }
[data-testid="stRadio"] label { align-items: center !important; }
[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important;
    color: #8a8069 !important;   /* unselected: legible warm grey */
}
[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #191510 !important; font-weight: 500 !important;   /* selected: bold + dark */
}
/* BaseWeb makes the <label> itself [data-baseweb=radio]; the circle is its first div child */
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    width: 18px !important; height: 18px !important; min-width: 18px !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    border: 2px solid #b8ad94 !important;   /* unselected: clear ring */
    box-shadow: none !important;
    transition: border-color .12s, border-width .12s;
}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child > div { display: none !important; }
/* Selected: bold navy ring (thick border, white centre) */
[data-testid="stRadio"] label:has(input:checked) > div:first-child {
    border: 5px solid var(--accent) !important;
    background: #ffffff !important;
}
/* Clearer hover affordance */
[data-testid="stRadio"] label[data-baseweb="radio"]:hover > div:first-child { border-color: var(--accent) !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] { cursor: pointer !important; }
/* Method-column vertical rhythm: 22px below the radios / uploader, before the button
   (applied to the flex item so it survives the container margin:0 reset above) */
[data-testid="stElementContainer"]:has(> [data-testid="stRadio"]) { margin-bottom: 22px !important; }
[data-testid="stElementContainer"]:has(> [data-testid="stFileUploader"]) { margin-bottom: 22px !important; }

/* ── Predict (primary) button ── */
/* Force the whole button chain to fill the column so the label centers with room to breathe */
[data-testid="stElementContainer"]:has(> .stButton) { width: 100% !important; }
.stButton { width: 100% !important; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    width: 100% !important;
    justify-content: center !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 20px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
.stButton > button[kind="primary"]:disabled { opacity: .55; }

/* ── Download-as-CSV rendered as a text link ── */
[data-testid="stElementContainer"]:has(> .stDownloadButton) { margin-top: 16px !important; }
[data-testid="stDownloadButton"] { display: inline-block !important; width: auto !important; }
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid var(--accent) !important;
    border-radius: 0 !important;
    padding: 0 0 2px !important;
    min-height: 0 !important;
    height: auto !important;
    width: auto !important;
    color: var(--accent) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    box-shadow: none !important;
    line-height: 1.4 !important;
}
[data-testid="stDownloadButton"] > button:hover { filter: brightness(1.15); }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: #faf9f6 !important;
    border: 1px dashed #d5ccb9 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
}

/* Serif labels / captions used across sections */
.dh-lbl { font-family: 'Spectral', serif; font-weight: 600; font-size: 16px; color: #191510; }
.dh-sub { font-size: 14px; color: #6b6252; margin: 4px 0 23px; }
.dh-cap { font: 12px 'IBM Plex Mono', monospace; color: #8a8069; margin-top: 9px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ───────────────────────────────────────────────────────────────
st.session_state.setdefault("results", None)   # list[{name, smiles, vals}] once a prediction runs
st.session_state.setdefault("notice", None)

# The dataset currently driving the table + chart — only populated once a prediction runs.
molecules = st.session_state["results"]


# ════════════════════════════════════════════════════════════════════════════════
# 1. HEADER
# ════════════════════════════════════════════════════════════════════════════════
pills = "".join(
    f'<span style="font:600 12px \'IBM Plex Mono\',monospace;color:#3a3428;'
    f'border:1px solid #d8cfbc;border-radius:999px;padding:6px 13px">{t}</span>'
    for t in TARGETS
)
st.markdown(
    f"""
<div style="padding:52px 0 0">
  <h1 style="font-family:'Spectral',serif;font-weight:600;font-size:42px;line-height:1.05;
             margin:0;letter-spacing:-.01em;color:#191510">REM Sleep Selectivity Profiler</h1>
  <p style="font-size:16px;line-height:1.55;color:#5b5344;margin:20px 0 0;white-space:nowrap">
     Predict binding affinity (pChEMBL) for small molecules across six CNS targets implicated in REM-sleep regulation.</p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 0">{pills}</div>
  <div style="height:2px;background:#d8cfbc;margin:30px 0 0"></div>
</div>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════════
# 2. INPUT + METHOD
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="height:34px"></div>', unsafe_allow_html=True)
col_in, col_method = st.columns([2.72, 1], gap="large")  # ~ 1fr / 300px

with col_in:
    st.markdown(
        '<h2 style="font-family:\'Spectral\',serif;font-weight:600;font-size:26px;margin:0;'
        'color:#191510">Input</h2><div class="dh-sub">one SMILES per line</div>',
        unsafe_allow_html=True)
    text = st.text_area(
        "Input", value="", height=150, label_visibility="collapsed",
        placeholder="Cc1ccc(-n2nccn2)c(C(=O)N2CCN(c3nc4cc(Cl)ccc4o3)CC[C@H]2C)c1",
    )
    smiles_raw = [s.strip() for s in text.splitlines() if s.strip()]
    st.markdown(f'<div class="dh-cap">{len(smiles_raw)} SMILES detected</div>', unsafe_allow_html=True)

with col_method:
    st.markdown('<div class="dh-lbl" style="margin-top:11px;padding-bottom:14px">Method</div>', unsafe_allow_html=True)
    method = st.radio("Method", ["Paste SMILES", "Upload .txt file"], label_visibility="collapsed")
    uploaded = None
    if method == "Upload .txt file":
        uploaded = st.file_uploader("Upload .txt (one SMILES per line)", type=["txt"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            smiles_raw = [s.strip() for s in uploaded.read().decode().splitlines() if s.strip()]
    run = st.button("Predict affinity", type="primary", disabled=len(smiles_raw) == 0)

# ── Run prediction ──────────────────────────────────────────────────────────────
if run and smiles_raw:
    canonical, invalid = validate_smiles(smiles_raw)
    if invalid:
        st.session_state["notice"] = ("warn", f"Skipped {len(invalid)} invalid SMILES: "
                                       + ", ".join(f"`{s}`" for s in invalid[:5])
                                       + ("…" if len(invalid) > 5 else ""))
    if not canonical:
        st.session_state["notice"] = ("error", "No valid SMILES to process.")
    else:
        try:
            from pathlib import Path
            missing = [t for t in TARGETS
                       if not (Path(__file__).parent / f"mlp_{t}.joblib").exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing model files: " + ", ".join(f"mlp_{t}.joblib" for t in missing))
            with st.spinner(f"Generating predictions for {len(canonical)} molecule(s)…"):
                tokenizer, encoder, mlps = get_models()
                from predict import predict as run_predict
                preds = run_predict(canonical, tokenizer, encoder, mlps)
            st.session_state["results"] = [
                {"name": f"Compound {i+1}", "smiles": smi,
                 "vals": [round(float(preds[t][i]), 3) for t in TARGETS]}
                for i, smi in enumerate(canonical)
            ]
            st.session_state["notice"] = None
        except Exception as e:  # models/deps not available in this environment
            st.session_state["notice"] = (
                "error",
                "Could not run the model in this environment "
                f"({type(e).__name__}: {e}). "
                "Ensure `mlp_<TARGET>.joblib` files and torch/transformers are installed.",
            )
    st.rerun()

if st.session_state["notice"]:
    level, msg = st.session_state["notice"]
    {"warn": st.warning, "error": st.error, "info": st.info}[level](msg)


# ════════════════════════════════════════════════════════════════════════════════
# 3. EXAMPLE COMPOUNDS  (iframe: clipboard copy + hover)
# ════════════════════════════════════════════════════════════════════════════════
examples_payload = [
    {**e, "short": trunc(e["smiles"], 44)} for e in EXAMPLES
]
examples_html = """
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{--accent:__ACCENT__}
  *{box-sizing:border-box}
  body{margin:0;font-family:'IBM Plex Sans',sans-serif;background:transparent}
  .hd{display:flex;align-items:baseline;justify-content:space-between}
  .hd h2{font-family:'Spectral',serif;font-weight:600;font-size:20px;margin:0;color:#191510}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}
  .card{cursor:pointer;border:1px solid #e6e0d5;border-radius:9px;padding:14px 16px;background:#faf9f6;transition:border-color .12s}
  .card:hover{border-color:var(--accent)}
  .row{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
  .name{font-family:'Spectral',serif;font-weight:600;font-size:15px;color:#191510}
  .lab{font:600 11px 'IBM Plex Mono',monospace;color:var(--accent)}
  .role{font:11px 'IBM Plex Mono',monospace;color:var(--accent);margin-top:3px}
  .smi{font:12px 'IBM Plex Mono',monospace;color:#5b5344;margin-top:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style></head>
<body>
  <div class="hd"><h2>Example compounds</h2></div>
  <div class="grid" id="grid"></div>
<script>
  const DATA = __DATA__;
  const grid = document.getElementById('grid');
  DATA.forEach((e, i) => {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<div class="row"><span class="name">'+e.name+'</span><span class="lab" id="lab'+i+'">Copy</span></div>'
      + '<div class="role">'+e.role+'</div><div class="smi">'+e.short+'</div>';
    c.onclick = () => {
      try { navigator.clipboard && navigator.clipboard.writeText(e.smiles); } catch(err) {}
      const lab = document.getElementById('lab'+i);
      lab.textContent = 'Copied ✓'; lab.style.color = '#2f9e63';
      clearTimeout(c._t);
      c._t = setTimeout(() => { lab.textContent = 'Copy'; lab.style.color = 'var(--accent)'; }, 1300);
    };
    grid.appendChild(c);
  });
  function fit(){ try{ const h=document.body.offsetHeight; const fe=window.frameElement;
    if(fe){ fe.style.height=h+'px'; if(fe.parentElement) fe.parentElement.style.height=h+'px'; } }catch(e){} }
  window.addEventListener('load', fit); setTimeout(fit, 60); setTimeout(fit, 300);
</script>
</body></html>
"""
examples_html = (examples_html
                 .replace("__ACCENT__", ACCENT)
                 .replace("__DATA__", json.dumps(examples_payload)))
st.markdown('<div style="height:36px"></div>', unsafe_allow_html=True)
components.html(examples_html, height=170, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════════
# 4. PREDICTED pChEMBL VALUES  (heatmap table, static markdown)
# ════════════════════════════════════════════════════════════════════════════════
if molecules:
    header_cells = (
        '<div style="padding:13px 18px;font:600 11px \'IBM Plex Mono\',monospace;letter-spacing:.06em;'
        'color:#6b6252;background:#f4f1ec">SMILES</div>'
        + "".join(
            f'<div style="padding:10px 8px;text-align:center;background:#f4f1ec;border-left:1px solid #e6e0d5">'
            f'<div style="font:600 11px \'IBM Plex Mono\',monospace;color:#3a3428">{t}</div>'
            f'<div style="font:400 9px \'IBM Plex Mono\',monospace;color:#8a8069;margin-top:2px">pChEMBL / IC50</div>'
            f'</div>'
            for t in TARGETS
        )
    )
    body_rows = ""
    for m in molecules:
        cells = (
            f'<div style="padding:15px 18px;font:500 12px \'IBM Plex Mono\',monospace;color:#443d30;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{trunc(m["smiles"], 34)}</div>'
        )
        for v in m["vals"]:
            ic50 = pchembl_to_ic50_str(v)
            cells += (
                f'<div style="padding:12px 8px;text-align:center;border-left:1px solid #ffffff;'
                f'background:{heat_bg(v)};color:{heat_fg(v)}">'
                f'<div style="font:600 14px \'IBM Plex Mono\',monospace">{v:.3f}</div>'
                f'<div style="font:500 10px \'IBM Plex Mono\',monospace;opacity:0.82;margin-top:3px">{ic50}</div>'
                f'</div>'
            )
        body_rows += (
            '<div style="display:grid;grid-template-columns:300px repeat(6,1fr);'
            f'border-top:1px solid #ece7dd;background:#ffffff">{cells}</div>'
        )

    table_html = """
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:'IBM Plex Sans',sans-serif;background:transparent}
  h2{font-family:'Spectral',serif;font-weight:600;font-size:26px;margin:0;color:#191510}
  p{font-size:14px;color:#6b6252;margin:8px 0 0}
  .tablewrap{margin-top:22px;border:1px solid #e6e0d5;border-radius:8px;overflow:hidden}
  .row{display:grid;grid-template-columns:300px repeat(6,1fr)}
</style></head>
<body>
  <h2>Predicted binding affinity</h2>
  <p>Color scale:
    <span style="color:#c9485b;font-weight:600">low</span> →
    <span style="color:#c98a1e;font-weight:600">mid</span> →
    <span style="color:#2f9e63;font-weight:600">high</span>. Each cell shows pChEMBL and IC50-equivalent (nM/µM/pM). IC50-equivalent is estimated as 10<sup>(9−pChEMBL)</sup> nM; pChEMBL aggregates IC50, Ki, Kd and EC50 measurements.</p>
  <div class="tablewrap">
    <div class="row">__HEADER_CELLS__</div>
    __BODY_ROWS__
  </div>
<script>
  function fit(){ try{ const h=document.body.offsetHeight; const fe=window.frameElement;
    if(fe){
      fe.style.height=h+'px'; fe.style.flexBasis=h+'px';
      if(fe.parentElement){ fe.parentElement.style.height=h+'px'; fe.parentElement.style.flexBasis=h+'px'; }
    } }catch(e){} }
  window.addEventListener('load', fit); setTimeout(fit, 60); setTimeout(fit, 300);
</script>
</body></html>
"""
    table_html = table_html.replace("__HEADER_CELLS__", header_cells).replace("__BODY_ROWS__", body_rows)
    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)
    components.html(table_html, height=100, scrolling=False)

    # CSV download: pChEMBL + IC50-equivalent columns
    ic50_headers = [f"{t}_IC50_nM" for t in TARGETS]
    csv_lines = ["SMILES," + ",".join(TARGETS) + "," + ",".join(ic50_headers)]
    for m in molecules:
        pchembl_cols = ",".join(f"{v:.3f}" for v in m["vals"])
        ic50_cols = ",".join(f"{10 ** (9 - v):.2f}" for v in m["vals"])
        csv_lines.append(m["smiles"] + "," + pchembl_cols + "," + ic50_cols)
    st.download_button("↓ Download as CSV", data="\n".join(csv_lines),
                       file_name="selectivity_predictions.csv", mime="text/csv")


    # ════════════════════════════════════════════════════════════════════════════════
    # 5. SELECTIVITY PROFILE  (iframe: Plotly grouped bars + custom fullscreen)
    # ════════════════════════════════════════════════════════════════════════════════
    chart_payload = {
        "targets": TARGETS,
        "colors": CCOLORS,
        "series": [{"name": trunc(m["smiles"], 34), "vals": m["vals"]} for m in molecules],
    }
    chart_html = """
    <!doctype html><html><head><meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      body{margin:0;font-family:'IBM Plex Sans',sans-serif;background:transparent}
      h2{font-family:'Spectral',serif;font-weight:600;font-size:26px;margin:0 0 6px;color:#191510}
      p{font-size:14px;color:#6b6252;margin:0 0 20px}
      #plot{width:100%;height:520px;background:#ffffff}
    </style></head>
    <body>
      <h2>Selectivity profile</h2>
      <p>Predicted pChEMBL per target, grouped by compound. Click a legend entry to hide it; hover the plot for the toolbar to zoom, pan, download, or go fullscreen.</p>
      <div id="plot"></div>
    <script>
      const D = __DATA__;
      const el = document.getElementById('plot');
      const tickFont = { family: "'IBM Plex Mono', monospace", size: 12, color: '#8a8069' };
      const titleFont = { family: "'IBM Plex Sans', sans-serif", size: 13, color: '#6b6252' };
      const legend = { title: { text: 'Molecule', font: { family: "'IBM Plex Sans', sans-serif", size: 12, color: '#6b6252' } },
                       font: { family: "'IBM Plex Mono', monospace", size: 11, color: '#5b5344' },
                       x: 1.01, y: 1, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(0,0,0,0)' };
      const traces = D.series.map((m, i) => ({
        type: 'bar', name: m.name, x: D.targets.slice(), y: m.vals.slice(),
        marker: { color: D.colors[i % D.colors.length] },
        text: m.vals.map(v => v.toFixed(2)), textposition: 'outside', cliponaxis: false,
        textfont: { family: "'IBM Plex Mono', monospace", size: 11, color: '#8a8069' },
        hovertemplate: '<b>%{x}</b>  pChEMBL %{y:.3f}<extra></extra>',
      }));
      const layout = {
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        font: { family: "'IBM Plex Sans', sans-serif", color: '#5b5344' },
        legend, hoverlabel: { font: { family: "'IBM Plex Mono', monospace" } },
        barmode: 'group', bargap: 0.35, bargroupgap: 0.12,
        margin: { l: 72, r: 20, t: 20, b: 60 },
        xaxis: { title: { text: 'Target', font: titleFont }, tickfont: tickFont, showgrid: false, showline: false, zeroline: false },
        yaxis: { title: { text: 'Predicted pChEMBL', font: titleFont }, tickfont: tickFont, range: [0, 10], gridcolor: '#ece7dd', zeroline: false, showline: false },
      };
      const fsIcon = { width: 448, height: 512, path: 'M0 180V56c0-13.3 10.7-24 24-24h124c6.6 0 12 5.4 12 12v40c0 6.6-5.4 12-12 12H64v84c0 6.6-5.4 12-12 12H12c-6.6 0-12-5.4-12-12zM288 44v40c0 6.6 5.4 12 12 12h84v84c0 6.6 5.4 12 12 12h40c6.6 0 12-5.4 12-12V56c0-13.3-10.7-24-24-24H300c-6.6 0-12 5.4-12 12zm148 240h-40c-6.6 0-12 5.4-12 12v84h-84c-6.6 0-12 5.4-12 12v40c0 6.6 5.4 12 12 12h124c13.3 0 24-10.7 24-24V296c0-6.6-5.4-12-12-12zM160 468v-40c0-6.6-5.4-12-12-12H64v-84c0-6.6-5.4-12-12-12H12c-6.6 0-12 5.4-12 12v124c0 13.3 10.7 24 24 24h124c6.6 0 12-5.4 12-12z' };
      const fsBtn = { name: 'fullscreen', title: 'Fullscreen', icon: fsIcon, click: (gd) => {
        if (!document.fullscreenElement) { const req = gd.requestFullscreen || gd.webkitRequestFullscreen || gd.msRequestFullscreen; if (req) req.call(gd); }
        else if (document.exitFullscreen) { document.exitFullscreen(); }
      }};
      const config = { displaylogo: false, responsive: true, modeBarButtonsToAdd: [fsBtn],
                       modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                       toImageButtonOptions: { filename: 'selectivity-profile', format: 'png', scale: 2 } };
      Plotly.newPlot(el, traces, layout, config);
      document.addEventListener('fullscreenchange', () => setTimeout(() => Plotly.Plots.resize(el), 60));
      window.addEventListener('resize', () => Plotly.Plots.resize(el));
    </script>
    </body></html>
    """
    chart_html = chart_html.replace("__DATA__", json.dumps(chart_payload))
    st.markdown('<div style="height:46px"></div>', unsafe_allow_html=True)
    components.html(chart_html, height=610, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════════
# 6. MODEL RELIABILITY & CAVEATS  (iframe: collapsible, self-sizing)
# ════════════════════════════════════════════════════════════════════════════════
reliab_payload = [{**r, "col": TIER_COL[r["tier"]]} for r in RELIAB]
reliab_html = """
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{--accent:__ACCENT__}
  *{box-sizing:border-box}
  body{margin:0;font-family:'IBM Plex Sans',sans-serif;background:transparent;color:#5b5344}
  .bar{cursor:pointer;background:#faf9f6;border:1px solid #e6e0d5;border-radius:11px;padding:20px 24px;display:flex;align-items:center;gap:14px}
  .caret{font:600 15px 'IBM Plex Mono',monospace;color:var(--accent);width:16px;text-align:center}
  .title{font-family:'Spectral',serif;font-weight:600;font-size:18px;color:#191510}
  .toglabel{margin-left:auto;font:11px 'IBM Plex Mono',monospace;color:#8a8069;letter-spacing:.08em;text-transform:uppercase}
  .panel{background:#fff;border:1px solid #e6e0d5;border-radius:11px;margin-top:10px;padding:26px 24px}
  .cap{font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.16em;text-transform:uppercase;color:#8a8069;margin-bottom:16px}
  .cards{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}
  .mc{border:1px solid #e6e0d5;border-radius:10px;padding:15px 14px;background:#faf9f6}
  .mt{font:600 12px 'IBM Plex Mono',monospace;color:#6b6252}
  .mr{font-family:'Spectral',serif;font-weight:600;font-size:28px;color:#191510;margin:8px 0 0}
  .track{height:5px;border-radius:3px;background:#ece7dd;overflow:hidden;margin:9px 0 8px}
  .fill{height:100%}
  .note{font:10.5px 'IBM Plex Mono',monospace}
  .rmse{font:10px 'IBM Plex Mono',monospace;color:#8a8069;margin-top:4px}
  .rule{height:1px;background:#ece7dd;margin:24px 0}
  .cv{display:flex;flex-direction:column;gap:11px}
  .cv .item{display:flex;gap:12px;font-size:13.5px;line-height:1.5;color:#5b5344}
  .cv .dash{color:var(--accent);font-family:'IBM Plex Mono',monospace}
</style></head>
<body>
  <div class="bar" id="bar">
    <span class="caret" id="caret">▸</span>
    <span class="title">Model reliability &amp; caveats</span>
    <span class="toglabel" id="toglabel">Expand</span>
  </div>
  <div class="panel" id="panel" style="display:none">
    <div class="cap">Pearson R · global CCPart split · 5-seed mean</div>
    <div class="cards" id="cards"></div>
    <div class="rule"></div>
    <div class="cap">Important caveats</div>
    <div class="cv" id="cv"></div>
  </div>
<script>
  const R = __RELIAB__, C = __CAVEATS__;
  const cards = document.getElementById('cards');
  R.forEach(r => {
    const d = document.createElement('div'); d.className = 'mc';
    d.innerHTML = '<div class="mt">'+r.t+'</div><div class="mr">'+r.r+'</div>'
      + '<div class="track"><div class="fill" style="width:'+r.w+';background:'+r.col+'"></div></div>'
      + '<div class="note" style="color:'+r.col+'">'+r.note+'</div>'
      + '<div class="rmse">RMSE '+r.rmse+'</div>';
    cards.appendChild(d);
  });
  const cv = document.getElementById('cv');
  C.forEach(t => {
    const d = document.createElement('div'); d.className = 'item';
    d.innerHTML = '<span class="dash">—</span><span>'+t+'</span>';
    cv.appendChild(d);
  });
  function fit(){ try{ const h=document.body.offsetHeight+2; const fe=window.frameElement;
    if(fe){ fe.style.height=h+'px'; if(fe.parentElement) fe.parentElement.style.height=h+'px'; } }catch(e){} }
  let open = false;
  const bar = document.getElementById('bar'), panel = document.getElementById('panel'),
        caret = document.getElementById('caret'), tog = document.getElementById('toglabel');
  bar.onclick = () => {
    open = !open;
    panel.style.display = open ? 'block' : 'none';
    caret.textContent = open ? '▾' : '▸';
    tog.textContent = open ? 'Collapse' : 'Expand';
    fit();
  };
  window.addEventListener('load', fit); setTimeout(fit, 60); window.addEventListener('resize', fit);
</script>
</body></html>
"""
reliab_html = (reliab_html
               .replace("__ACCENT__", ACCENT)
               .replace("__RELIAB__", json.dumps(reliab_payload))
               .replace("__CAVEATS__", json.dumps(CAVEATS)))
st.markdown('<div style="height:46px"></div>', unsafe_allow_html=True)
components.html(reliab_html, height=90, scrolling=False)
