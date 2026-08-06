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

    skip_embeddings = st.checkbox("Skip embedding analysis", value=False)

    max_embed = st.slider(
        "Max solutions to embed",
        min_value=100, max_value=2000, value=500, step=50,
        disabled=skip_embeddings,
        help="500 solutions ≈ 10 min on CPU. Enough to prove the orthogonality claim.",
    )

    output_dir_str = st.text_input("Output directory", value="results")

    st.divider()
    run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

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
if run_btn:
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)

    from config import Config
    cfg = Config()
    cfg.embedding_max_solutions = max_embed

    # ---- Phase 1 ----
    with st.status("Phase 1 — Loading NeoCoder dataset …", expanded=True) as phase1:
        from data.loader import load_dataset
        solutions, gpt4_labels = load_dataset(cfg)
        n_probs = len(solutions)
        n_sols  = sum(len(v) for v in solutions.values())
        phase1.update(
            label=f"Phase 1 — {n_probs} problems · {n_sols} Python solutions",
            state="complete", expanded=False,
        )

    # ---- Phase 2 ----
    ast_labels_path = output_dir / "ast_labels.json"
    with st.status("Phase 2 — Deterministic AST parsing …", expanded=True) as phase2:
        from ast_analysis.visitor import ASTAnalyzer
        if skip_ast and ast_labels_path.exists():
            ast_labels = ASTAnalyzer.load(ast_labels_path)
            phase2.update(
                label="Phase 2 — AST labels loaded from cache",
                state="complete", expanded=False,
            )
        else:
            analyzer = ASTAnalyzer(cfg.techniques)
            ast_labels = analyzer.analyze_all(solutions)
            analyzer.save(ast_labels, ast_labels_path)
            phase2.update(
                label="Phase 2 — AST parsing complete (100% accurate ground truth)",
                state="complete", expanded=False,
            )

    # ---- Phase 3 ----
    llm_labels_path = output_dir / "llm_labels.json"
    with st.status("Phase 3 — LLM Judge evaluation …", expanded=True) as phase3:
        if skip_llm:
            from llm_judge.evaluator import convert_gpt4_labels
            llm_labels = convert_gpt4_labels(
                gpt4_labels, cfg.techniques, cfg.technique_label_map
            )
            llm_labels_path.write_text(json.dumps(llm_labels, indent=2), encoding="utf-8")
            phase3.update(
                label="Phase 3 — GPT-4 labels converted (no API call)",
                state="complete", expanded=False,
            )
        else:
            if llm_labels_path.exists():
                from llm_judge.evaluator import LLMJudge
                llm_labels = LLMJudge.load(llm_labels_path)
                phase3.update(
                    label="Phase 3 — Llama labels loaded from cache",
                    state="complete", expanded=False,
                )
            else:
                from llm_judge.evaluator import LLMJudge
                judge = LLMJudge(cfg)
                llm_labels = judge.evaluate_all(solutions)
                judge.save(llm_labels, llm_labels_path)
                phase3.update(
                    label=f"Phase 3 — Llama-3-8B judged {sum(len(v) for v in llm_labels.values())} solutions",
                    state="complete", expanded=False,
                )

    from llm_judge.evaluator import convert_gpt4_labels
    gpt4_binary = convert_gpt4_labels(
        gpt4_labels, cfg.techniques, cfg.technique_label_map
    )
    (output_dir / "gpt4_binary_labels.json").write_text(json.dumps(gpt4_binary, indent=2), encoding="utf-8")

    # ---- Phase 4 ----
    embedding_results = None
    emb_path = output_dir / "embedding_results.json"

    if skip_embeddings:
        st.info("Phase 4 — Embedding analysis skipped.")
    else:
        with st.status("Phase 4 — DeBERTa embedding analysis …", expanded=True) as phase4:
            from embeddings.analyzer import EmbeddingAnalyzer
            if emb_path.exists():
                embedding_results = EmbeddingAnalyzer.load(emb_path)
                phase4.update(
                    label="Phase 4 — Embedding results loaded from cache",
                    state="complete", expanded=False,
                )
            else:
                prog_bar  = st.progress(0.0)
                prog_text = st.empty()

                def _encoding_progress(frac: float):
                    prog_bar.progress(frac)
                    prog_text.caption(f"Encoding solutions … {frac:.0%}")

                emb_analyzer = EmbeddingAnalyzer(cfg)
                results_obj = emb_analyzer.analyze(
                    solutions, ast_labels, progress_callback=_encoding_progress
                )
                emb_analyzer.save(results_obj, emb_path)
                embedding_results = results_obj.to_dict()

                prog_bar.progress(1.0)
                prog_text.empty()
                phase4.update(
                    label=f"Phase 4 — Encoding done · Pearson r = {results_obj.correlation:.3f}",
                    state="complete", expanded=False,
                )

    # ---- Phase 5 ----
    with st.status("Phase 5 — Computing metrics & generating report …", expanded=False) as phase5:
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
        phase5.update(
            label="Phase 5 — Report generated ✓",
            state="complete", expanded=False,
        )

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
                       use_container_width=True)
        if gpt4_png.exists():
            col2.image(str(gpt4_png), caption="GPT-4 vs. AST Ground Truth",
                       use_container_width=True)
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
            c1.dataframe(df, use_container_width=True, hide_index=True)
        if gpt4_csv.exists():
            df = pd.read_csv(gpt4_csv)[cols].round(3)
            c2.subheader("GPT-4")
            c2.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("FPR = hallucination rate · FNR = miss rate")

    # ---- Embedding analysis ----
    with tab_emb:
        emb_path = results_dir / "embedding_results.json"
        if emb_path.exists():
            emb = json.loads(emb_path.read_text(encoding="utf-8"))
            r = emb.get("correlation")
            pairs = emb.get("divergent_pairs", [])

            m1, m2, m3 = st.columns(3)
            m1.metric("Pearson r  (cosine vs structural sim)", f"{r:.3f}" if r is not None else "N/A")
            m2.metric("Divergent pairs found", len(pairs))
            m3.metric("Solutions embedded", emb.get("n_solutions", "—"))

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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run the pipeline with embedding analysis enabled to see results here.")

    # ---- Full report ----
    with tab_report:
        st.markdown(report_path.read_text(encoding="utf-8"))
        st.divider()
        st.download_button(
            "⬇️ Download report.md",
            data=report_path.read_text(encoding="utf-8"),
            file_name="report.md",
            mime="text/markdown",
        )
