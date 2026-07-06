"""SovereignLex Streamlit UI — Explainable Multi-Agent Legal Reasoning Workbench.

Usage:
    streamlit run sovereignlex/app.py
"""

import os
import time

import streamlit as st

from sovereignlex.models import ClaimInput, ClaimType, PartyRole
from sovereignlex.orchestrator import Orchestrator
from sovereignlex.llm import LLMConfig

st.set_page_config(
    page_title="SovereignLex — Legal Reasoning Workbench",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIVACY_BANNER = (
    "OpenCaseLaw API used for legal reference retrieval only. "
    "Select Ollama for fully local/sovereign processing."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ SovereignLex")
    st.caption("Explainable Multi-Agent Legal Reasoning Workbench")
    st.markdown("---")

    st.subheader("LLM Configuration")
    provider = st.selectbox(
        "Provider",
        options=["deepseek", "ollama", "openai"],
        index=0,
        format_func=lambda x: {"deepseek": "DeepSeek API (recommended)", "ollama": "Ollama (local)", "openai": "OpenAI API"}[x],
    )
    if provider == "deepseek":
        model = st.selectbox(
            "Model",
            options=["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
            format_func=lambda x: "DeepSeek V3/V4 (Chat)" if "chat" in x else "DeepSeek R1 (Reasoner)",
        )
        api_key = st.text_input("DeepSeek API Key", type="password",
                                value=os.environ.get("DEEPSEEK_API_KEY", ""),
                                help="Set DEEPSEEK_API_KEY env var or paste here")
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
    elif provider == "openai":
        model = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"])
        api_key = st.text_input("OpenAI API Key", type="password",
                                value=os.environ.get("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        model = st.selectbox("Model", options=["llama3.1:8b", "qwen2.5-coder:32b", "gemma3:4b"])
        api_key = None

    st.markdown("---")

    st.subheader("Select Example")
    example_idx = st.selectbox(
        "Pre-loaded tenancy case",
        options=list(range(5)),
        format_func=lambda i: [
            "1. Termination Without Official Form",
            "2. Rent Increase After Renovation",
            "3. Defective Heating in Winter",
            "4. Retaliatory Termination",
            "5. Deposit Not Returned",
        ][i],
    )

    claimant_role = st.selectbox(
        "Claimant role",
        options=["tenant", "landlord"],
        index=0,
    )

    claim_type = st.selectbox(
        "Claim type",
        options=["termination_validity", "rent_increase", "defect_remediation", "deposit_dispute", "other"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    language = st.selectbox("Language", options=["de", "fr", "it"], index=0)

    st.markdown("---")
    st.caption(PRIVACY_BANNER)

# ── Main Content ────────────────────────────────────────────────────────────

st.title("SovereignLex Workbench")
st.markdown(
    "A transparent, modular multi-agent reasoning architecture for Swiss Tenancy Law (Mietrecht). "
    "This workbench demonstrates how explainable AI can assist legal experts — "
    "**not replace them**."
)

col1, col2 = st.columns([3, 1])
with col1:
    claim_text = st.text_area(
        "Enter your claim or dispute description",
        height=200,
        placeholder="Describe the tenancy dispute in detail...",
    )
with col2:
    st.caption("Or load a pre-built example")
    if st.button("Load Example", use_container_width=True):
        from examples.tenancy_cases import EXAMPLES
        ex = EXAMPLES[example_idx]
        st.session_state.claim_text = ex["raw_text"]
        st.rerun()

if "claim_text" not in st.session_state:
    st.session_state.claim_text = ""

run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Analyze Claim", type="primary", use_container_width=True, disabled=not st.session_state.claim_text.strip())

# ── Results Section ─────────────────────────────────────────────────────────

if run_clicked or "pipeline_result" in st.session_state:
    if run_clicked:
        with st.spinner("Running multi-agent reasoning pipeline..."):
            config = LLMConfig(provider=provider, model=model)
            if api_key:
                config.api_key = api_key
            orchestrator = Orchestrator(llm_config=config)
            try:
                claim = ClaimInput(
                    raw_text=st.session_state.claim_text,
                    claimant_role=PartyRole(claimant_role),
                    claim_type=ClaimType(claim_type),
                    canton="CH",
                    language=language,
                )
                result = orchestrator.run(claim)
                st.session_state.pipeline_result = result
            finally:
                orchestrator.close()

    result = st.session_state.get("pipeline_result")

    if result is None:
        st.stop()

    st.markdown("---")

    # ── Reasoning Trace ────────────────────────────────────────────────────

    st.subheader("🔍 Reasoning Trace")
    st.caption("Complete agent execution log with step-by-step transparency")

    trace_cols = st.columns([0.5, 2, 2, 0.5, 3])
    trace_cols[0].markdown("**Status**")
    trace_cols[1].markdown("**Agent**")
    trace_cols[2].markdown("**Confidence**")
    trace_cols[3].markdown("**Time**")
    trace_cols[4].markdown("**Summary**")

    for step in result.trace:
        icon = "✅" if step.status == "completed" else "❌"
        cols = st.columns([0.5, 2, 2, 0.5, 3])
        cols[0].markdown(icon)
        cols[1].markdown(f"**{step.agent_name}**")
        cols[2].markdown(f"{step.confidence}%" if step.confidence else "-")
        cols[3].markdown(f"{step.duration_ms}ms")
        cols[4].caption(step.output_summary[:120])

    st.caption(f"Total pipeline duration: {result.total_duration_ms}ms | Status: {result.status}")

    if result.error:
        st.error(f"Pipeline error: {result.error}")
        st.stop()

    # ── Final Report ───────────────────────────────────────────────────────

    if result.synthesis_report:
        report = result.synthesis_report
        st.markdown("---")

        # Confidence gauge
        st.subheader("📊 Confidence Assessment")
        conf_col1, conf_col2, conf_col3 = st.columns([1, 2, 1])
        with conf_col1:
            st.metric("Overall Confidence", f"{report.confidence_score}%")
        with conf_col2:
            st.progress(report.confidence_score / 100.0)
            st.caption(report.confidence_justification[:200])
        with conf_col3:
            if report.confidence_breakdown:
                bd = report.confidence_breakdown
                st.caption(f"Fact Quality: {bd.fact_quality}%")
                st.caption(f"Legal Basis: {bd.legal_basis_strength}%")
                st.caption(f"Precedent Alignment: {bd.precedent_alignment}%")

        st.markdown("---")

        # Executive Summary
        st.subheader("📋 Executive Summary")
        st.info(report.executive_summary)

        # Key findings + Leaning conclusion
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.subheader("🔑 Key Findings")
            if report.key_findings:
                for f in report.key_findings:
                    st.markdown(f"- {f}")
            else:
                st.caption("No key findings generated.")
        with fcol2:
            st.subheader("⚖️ Leaning Conclusion")
            if report.leaning_conclusion:
                st.warning(report.leaning_conclusion)
            else:
                st.caption("No conclusion reached.")

        st.markdown("---")

        # Retrieved Precedents
        st.subheader("📚 Key Precedents & Statutes")
        if report.key_precedents:
            for p in report.key_precedents[:5]:
                cite = p.get("citation", p.get("citation_string", "N/A"))
                url = p.get("url", p.get("canonical_url", ""))
                relevance = p.get("relevance", p.get("relevance_explanation", ""))
                if url:
                    st.markdown(f"- [{cite}]({url}) — {relevance[:200]}")
                else:
                    st.markdown(f"- **{cite}** — {relevance[:200]}")
        elif result.retrieval_result:
            for c in result.retrieval_result.cases_retrieved[:5]:
                url = c.canonical_url
                cite = c.citation_string
                if url:
                    st.markdown(f"- [{cite}]({url})")
                else:
                    st.markdown(f"- **{cite}**")

        if report.applicable_statutes:
            with st.expander("Applicable Statutes"):
                for s in report.applicable_statutes[:5]:
                    st.markdown(f"**{s.get('article', 'N/A')}** — {s.get('text', '')[:300]}")

        st.markdown("---")

        # Evidence Gaps
        st.subheader("⚠️ Evidence Gaps")
        if report.evidence_gaps:
            for g in report.evidence_gaps:
                desc = g.description if hasattr(g, 'description') else str(g)
                imp = g.importance if hasattr(g, 'importance') else ""
                if desc.strip():
                    st.markdown(f"- **{desc}** — {imp}")
        elif result.evidence_mapping and result.evidence_mapping.gaps:
            for g in result.evidence_mapping.gaps:
                if g.description.strip():
                    st.markdown(f"- **{g.description}** — {g.importance}")
        else:
            st.caption("No significant evidence gaps identified.")

        # Counterarguments
        st.subheader("⚡ Counterarguments (Opposing View)")
        if report.counterarguments:
            for ca in report.counterarguments:
                strength = ca.strength if hasattr(ca, 'strength') else "moderate"
                arg = ca.argument if hasattr(ca, 'argument') else str(ca)
                legal = ca.legal_basis if hasattr(ca, 'legal_basis') else ""
                with st.expander(f"[{strength.upper()}] {arg[:100]}..."):
                    st.markdown(f"**Argument:** {arg}")
                    if legal:
                        st.markdown(f"**Legal Basis:** {legal}")
                    if hasattr(ca, 'rebuttal') and ca.rebuttal:
                        st.markdown(f"**Possible Rebuttal:** {ca.rebuttal}")
        else:
            st.caption("No counterarguments generated.")

        # Recommended Actions
        if report.recommended_actions:
            st.subheader("📝 Recommended Actions")
            for action in report.recommended_actions:
                act_text = action.action if hasattr(action, 'action') else str(action)
                priority = action.priority if hasattr(action, 'priority') else "medium"
                rationale = action.rationale if hasattr(action, 'rationale') else ""
                st.markdown(f"- **[{priority.upper()}]** {act_text}")
                if rationale:
                    st.caption(f"  {rationale}")

        st.markdown("---")

        # Raw outputs in expanders
        with st.expander("Raw Analysis Plan (Coordinator Output)"):
            if result.analysis_plan:
                st.json(result.analysis_plan.model_dump())

        with st.expander("Raw Extracted Facts"):
            if result.extracted_facts:
                st.json(result.extracted_facts.model_dump(mode="json"))

        with st.expander("Raw Verification Report"):
            if result.verification_report:
                st.json(result.verification_report.model_dump())

    # ── Disclaimer ──────────────────────────────────────────────────────────

    st.markdown("---")
    st.caption(
        "⚠️ **Research Prototype Disclaimer:** This is a research and decision-support "
        "prototype for exploring precedents and reasoning transparency. "
        "It is **NOT legal advice**. All conclusions must be reviewed by a qualified "
        "lawyer. Outputs may contain errors. Designed to demonstrate transparent, "
        "sovereign AI reasoning methods."
    )
