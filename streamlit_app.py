"""
Streamlit dashboard for the AST vs. LLM-as-a-Judge benchmark.

Run with:
    streamlit run streamlit_app.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AST vs. LLM-as-a-Judge",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    skip_llm = st.checkbox(
        "Use GPT-4 labels (no API call)",
        value=True,
        help="Uncheck to call Llama-3-8B via Groq — requires GROQ_API_KEY in .env",
    )

    ast_cache_exists = Path("results/ast_labels.json").exists()
    skip_ast = st.checkbox(
        "Reuse cached AST labels",
        value=ast_cache_exists,
        disabled=not ast_cache_exists,
        help="Available only when results/ast_labels.json already exists",
    )

    max_llm = st.slider(
        "Max solutions for LLM judge",
        min_value=10, max_value=300, value=50, step=10,
        disabled=skip_llm,
        help="Each Groq call takes a few seconds on the free tier. 50 ≈ a few min.",
    )

    skip_embeddings = st.checkbox("Skip embedding analysis", value=False)

    max_embed = st.slider(
        "Max solutions to embed",
        min_value=100, max_value=2000, value=500, step=50,
        disabled=skip_embeddings,
        help="500 solutions ≈ 10 min on CPU. Enough to prove the orthogonality claim.",
    )

    output_dir_str = st.text_input("Output directory", value="results")

    st.divider()
    run_btn = st.button("▶  Run Pipeline", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔬 Benchmarking the Evaluators")
st.markdown(
    "**AST vs. LLM-as-a-Judge** &nbsp;·&nbsp; "
    "Amit Saxena &nbsp;·&nbsp; Saarland University / MPI-SWS"
)
st.divider()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def make_phase(title: str):
    """Render a phase row with a label + progress bar + status caption.

    Returns (set_progress, finish) callables.
    """
    label = st.markdown(f"**{title}**")
    bar = st.progress(0.0)
    caption = st.empty()

    def set_progress(frac: float, msg: str = ""):
        bar.progress(min(max(frac, 0.0), 1.0))
        if msg:
            caption.caption(msg)

    def finish(msg: str):
        bar.progress(1.0)
        caption.caption(f"✅ {msg}")

    return set_progress, finish


if run_btn:
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)

    from config import Config
    cfg = Config()
    cfg.embedding_max_solutions = max_embed
    cfg.llm_max_solutions = max_llm

    # ---- Phase 1 — Load dataset ----
    p1_set, p1_done = make_phase("Phase 1 · Loading NeoCoder dataset")
    from data.loader import load_dataset
    p1_set(0.3, "Reading dataset files …")
    solutions, gpt4_labels = load_dataset(cfg)
    n_probs = len(solutions)
    n_sols  = sum(len(v) for v in solutions.values())
    p1_done(f"{n_probs} problems · {n_sols} Python solutions")

    # ---- Phase 2 — AST parsing ----
    p2_set, p2_done = make_phase("Phase 2 · Deterministic AST parsing (ground truth)")
    ast_labels_path = output_dir / "ast_labels.json"
    from ast_analysis.visitor import ASTAnalyzer
    if skip_ast and ast_labels_path.exists():
        ast_labels = ASTAnalyzer.load(ast_labels_path)
        p2_done("AST labels loaded from cache")
    else:
        analyzer = ASTAnalyzer(cfg.techniques)
        ast_labels = analyzer.analyze_all(
            solutions,
            progress_callback=lambda f: p2_set(f, f"Parsing syntax trees … {f:.0%}"),
        )
        analyzer.save(ast_labels, ast_labels_path)
        p2_done("AST parsing complete (100% accurate ground truth)")

    # ---- Phase 3 — LLM judge ----
    p3_set, p3_done = make_phase("Phase 3 · LLM-as-a-Judge evaluation")
    llm_labels_path = output_dir / "llm_labels.json"
    if skip_llm:
        from llm_judge.evaluator import convert_gpt4_labels
        p3_set(0.5, "Converting GPT-4 labels (no API call) …")
        llm_labels = convert_gpt4_labels(gpt4_labels, cfg.techniques, cfg.technique_label_map)
        llm_labels_path.write_text(json.dumps(llm_labels, indent=2))
        p3_done("GPT-4 labels converted (no API call)")
    elif llm_labels_path.exists():
        from llm_judge.evaluator import LLMJudge
        llm_labels = LLMJudge.load(llm_labels_path)
        p3_done("Llama labels loaded from cache (delete results/llm_labels.json to re-run)")
    else:
        from llm_judge.evaluator import LLMJudge
        judge = LLMJudge(cfg)
        llm_labels = judge.evaluate_all(
            solutions, ast_labels=ast_labels,
            progress_callback=lambda d, t: p3_set(d / t, f"Querying Llama via Groq … {d}/{t} solutions"),
        )
        judge.save(llm_labels, llm_labels_path)
        p3_done(f"Llama-3.1 judged {sum(len(v) for v in llm_labels.values())} solutions")

    from llm_judge.evaluator import convert_gpt4_labels
    gpt4_binary = convert_gpt4_labels(gpt4_labels, cfg.techniques, cfg.technique_label_map)
    (output_dir / "gpt4_binary_labels.json").write_text(json.dumps(gpt4_binary, indent=2))

    # ---- Phase 4 — Embeddings ----
    embedding_results = None
    emb_path = output_dir / "embedding_results.json"
    if skip_embeddings:
        p4_set, p4_done = make_phase("Phase 4 · DeBERTa embedding analysis")
        p4_done("Skipped")
    else:
        p4_set, p4_done = make_phase("Phase 4 · DeBERTa embedding analysis")
        from embeddings.analyzer import EmbeddingAnalyzer
        if emb_path.exists():
            embedding_results = EmbeddingAnalyzer.load(emb_path)
            p4_done("Embedding results loaded from cache")
        else:
            emb_analyzer = EmbeddingAnalyzer(cfg)
            results_obj = emb_analyzer.analyze(
                solutions, ast_labels,
                progress_callback=lambda f: p4_set(f, f"Encoding with DeBERTa (CPU) … {f:.0%}"),
            )
            emb_analyzer.save(results_obj, emb_path)
            embedding_results = results_obj.to_dict()
            p4_done(f"Encoding done · Pearson r = {results_obj.correlation:.3f}")

    # ---- Phase 5 — Report ----
    p5_set, p5_done = make_phase("Phase 5 · Computing metrics & generating report")
    p5_set(0.4, "Computing metrics …")
    from reporting.metrics import MetricsReporter
    reporter = MetricsReporter(cfg)
    reporter.generate(
        solutions=solutions,
        ast_labels=ast_labels,
        llm_labels=llm_labels,
        gpt4_labels=gpt4_binary,
        embedding_results=embedding_results,
        output_dir=output_dir,
    )
    p5_done("Report generated")

    st.success(f"Pipeline complete. Outputs saved to `{output_dir.resolve()}`")
    st.session_state["output_dir"] = output_dir_str

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
results_dir = Path(st.session_state.get("output_dir", "results"))
report_path = results_dir / "report.md"

if report_path.exists():
    st.header("📊 Results")

    tab_heat, tab_metrics, tab_emb, tab_report = st.tabs([
        "🗺️ Heatmaps",
        "📈 Metrics",
        "🔗 Embedding Analysis",
        "📄 Full Report",
    ])

    # ---- Heatmaps ----
    with tab_heat:
        col1, col2 = st.columns(2)
        llm_png  = results_dir / "llm_heatmap.png"
        gpt4_png = results_dir / "gpt4_heatmap.png"
        if llm_png.exists():
            col1.image(str(llm_png), caption="Llama-3-8B vs. AST Ground Truth",
                       width="stretch")
        if gpt4_png.exists():
            col2.image(str(gpt4_png), caption="GPT-4 vs. AST Ground Truth",
                       width="stretch")
        st.caption(
            "**Dark red = poor performance.** "
            "High accuracy paired with F1 = 0 reveals class-imbalance masking. "
            "F1 = 0 for `lambda` and `list_comprehension` means the LLM never detected them."
        )

    # ---- Metrics table ----
    with tab_metrics:
        llm_csv  = results_dir / "llm_metrics.csv"
        gpt4_csv = results_dir / "gpt4_metrics.csv"
        cols = ["technique", "accuracy", "precision", "recall", "f1", "fpr", "fnr"]

        c1, c2 = st.columns(2)
        if llm_csv.exists():
            df = pd.read_csv(llm_csv)[cols].round(3)
            c1.subheader("Llama-3-8B-Instruct")
            c1.dataframe(df, width="stretch", hide_index=True)
        if gpt4_csv.exists():
            df = pd.read_csv(gpt4_csv)[cols].round(3)
            c2.subheader("GPT-4")
            c2.dataframe(df, width="stretch", hide_index=True)

        st.caption("FPR = hallucination rate · FNR = miss rate")

    # ---- Embedding analysis ----
    with tab_emb:
        emb_path = results_dir / "embedding_results.json"
        if emb_path.exists():
            emb = json.loads(emb_path.read_text())
            r = emb.get("correlation")
            pairs = emb.get("divergent_pairs", [])

            n_pairs = len(emb.get("pairwise_data", []))
            m1, m2, m3 = st.columns(3)
            m1.metric("Pearson r  (cosine vs structural sim)", f"{r:.3f}" if r is not None else "N/A")
            m2.metric("Divergent pairs found", len(pairs))
            m3.metric("Pair comparisons evaluated", f"{n_pairs:,}")

            if r is not None and abs(r) < 0.1:
                st.success(
                    f"r = {r:.3f} ≈ 0 confirms the core hypothesis: "
                    "semantic similarity is orthogonal to structural compliance."
                )

            if pairs:
                st.subheader("Top Divergent Pairs")
                st.caption(
                    "These solution pairs are semantically near-identical (high cosine sim) "
                    "but structurally divergent (different AST techniques) — proving embeddings "
                    "cannot enforce structural constraints."
                )
                rows = [
                    {
                        "#": i + 1,
                        "Problem A": f"{p['pid_a']}[{p['idx_a']}]",
                        "Problem B": f"{p['pid_b']}[{p['idx_b']}]",
                        "Cosine Sim": round(p["cosine_similarity"], 3),
                        "Differing Techniques": ", ".join(p["differing_techniques"]),
                    }
                    for i, p in enumerate(pairs)
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("Run the pipeline with embedding analysis enabled to see results here.")

    # ---- Full report ----
    with tab_report:
        st.markdown(report_path.read_text())
        st.divider()
        st.download_button(
            "⬇️ Download report.md",
            data=report_path.read_text(),
            file_name="report.md",
            mime="text/markdown",
        )
