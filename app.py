"""SovereignLex Streamlit UI — Polished Demo for PhD Committee.

Usage:
    streamlit run app.py
"""

import os
import time

import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="SovereignLex — Multi-Agent Legal Reasoning",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark theme with blue/gold accents — legal/professional feel */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .card-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .gap-warning {
        background: linear-gradient(135deg, #451a03, #78350f);
        border: 1px solid #92400e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #fbbf24;
    }
    .counter-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #4338ca;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .counter-card .strength {
        color: #818cf8;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    .counter-card .argument {
        color: #e2e8f0;
        margin-top: 0.3rem;
    }
    .confidence-high { color: #22c55e; }
    .confidence-medium { color: #fbbf24; }
    .confidence-low { color: #ef4444; }
    .finding-item {
        background: #1e293b;
        border-left: 3px solid #60a5fa;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    .statute-ref {
        font-family: 'JetBrains Mono', monospace;
        background: #334155;
        color: #fbbf24;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    hr {
        border-color: #334155;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Imports ──────────────────────────────────────────────────────────────────

from sovereignlex.models import ClaimInput, ClaimType, PartyRole
from sovereignlex.orchestrator import Orchestrator
from sovereignlex.llm import LLMConfig

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=50) if False else None
    st.markdown("## ⚖️ SovereignLex")
    st.caption("Explainable Multi-Agent Legal Reasoning")
    st.markdown("---")

    st.subheader("🔧 Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["deepseek", "ollama", "openai"],
        index=0,
        format_func=lambda x: {
            "deepseek": "DeepSeek API (fast, best quality)",
            "ollama": "Ollama (local, sovereign)",
            "openai": "OpenAI API",
        }[x],
        help="Choose your AI backend. DeepSeek is recommended for quality and speed.",
    )

    if provider == "deepseek":
        model = st.selectbox(
            "Model",
            options=["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
            format_func=lambda x: "⚡ DeepSeek Chat (V3/V4)" if "chat" in x else "🧠 DeepSeek Reasoner (R1)",
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            placeholder="sk-...",
            help="Get your key at platform.deepseek.com",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
    elif provider == "openai":
        model = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"])
        api_key = st.text_input("API Key", type="password",
                                value=os.environ.get("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        model = st.selectbox(
            "Model",
            options=["llama3.1:8b", "qwen2.5-coder:32b"],
            help="Requires Ollama running locally",
        )
        api_key = None

    st.markdown("---")
    st.markdown("<small style='color: #64748b;'>OpenCaseLaw API used for legal data retrieval only.</small>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_demo, tab_manual, tab_about = st.tabs(["🎯 One-Click Demo", "📝 Custom Analysis", "ℹ️ About"])

# ── Tab 1: One-Click Demo ───────────────────────────────────────────────────

with tab_demo:
    st.markdown('<div class="main-header">SovereignLex</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A transparent multi-agent reasoning workbench for Swiss Tenancy Law</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Select a pre-built Swiss tenancy dispute")
    st.caption("Each case includes realistic facts, automatically retrieves real BGE court decisions, and generates a full reasoning report with counterarguments.")

    examples = [
        {
            "title": "📋 Termination Without Official Form",
            "desc": "Landlord sends a simple letter instead of the required cantonal form. Tenant challenges validity under Art. 266l OR.",
            "laws": "Art. 266l OR, Art. 271 OR",
        },
        {
            "title": "💰 Rent Increase After Renovation",
            "desc": "Kitchen renovation leads to a 20% rent hike. Tenant questions whether the increase is justified under Art. 269 OR.",
            "laws": "Art. 269 OR, Art. 269d OR",
        },
        {
            "title": "🌡️ Defective Heating in Winter",
            "desc": "Heating fails, apartment drops below 18°C. Tenant withholds rent. Landlord threatens eviction. Was the rent reduction legal?",
            "laws": "Art. 259a OR, Art. 259d OR, Art. 259g OR",
        },
        {
            "title": "⚡ Retaliatory Termination",
            "desc": "Tenant files a defect complaint. 15 days later, landlord terminates the lease. Coincidence or revenge under Art. 271a OR?",
            "laws": "Art. 271a OR, Art. 271 OR",
        },
        {
            "title": "🔒 Deposit Not Returned",
            "desc": "Six months after moving out, the rental deposit is still blocked. Landlord claims 'hidden damage' without proof. Does the tenant have recourse?",
            "laws": "Art. 257e OR",
        },
    ]

    cols = st.columns([1, 1, 1])
    selected_example = None

    for i, ex in enumerate(examples):
        col_idx = i % 3
        if i < 3:
            c = cols[i]
        elif i == 3:
            cols = st.columns([1, 1])
            c = cols[0]
        else:
            c = cols[1]

        with c:
            with st.container():
                st.markdown(f"**{ex['title']}**")
                st.caption(ex["desc"])
                st.markdown(f'<span class="statute-ref">{ex["laws"]}</span>', unsafe_allow_html=True)
                if st.button(f"Analyze Case {i+1}", key=f"demo_{i}", use_container_width=True):
                    selected_example = i

    st.markdown("---")

    if selected_example is not None:
        run_demo = True
    else:
        run_demo = False
        st.info("👆 Click any 'Analyze Case' button above to see the multi-agent reasoning pipeline in action.")

    if run_demo:
        from examples.tenancy_cases import EXAMPLES

        ex = EXAMPLES[selected_example]
        with st.spinner("🤖 Running multi-agent reasoning pipeline... This may take 30-60 seconds."):
            config = LLMConfig(provider=provider, model=model)
            if api_key:
                config.api_key = api_key
            orchestrator = Orchestrator(llm_config=config)
            try:
                claim = ClaimInput(
                    raw_text=ex["raw_text"],
                    claimant_role=PartyRole(ex.get("claimant_role", "tenant")),
                    claim_type=ClaimType(ex.get("claim_type", "termination_validity")),
                    canton=ex.get("canton", "CH"),
                    language=ex.get("language", "de"),
                    attached_documents=ex.get("attached_documents", []),
                )
                result = orchestrator.run(claim)
                st.session_state.demo_result = result
                st.session_state.demo_name = ex["name"]
            finally:
                orchestrator.close()

    if "demo_result" in st.session_state:
        result = st.session_state.demo_result
        st.success(f"✅ Analysis complete: **{st.session_state.demo_name}**")

        # ── Results Layout ──────────────────────────────────────────────────

        # Row 1: Key metrics
        st.markdown("### 📊 Analysis Overview")
        m1, m2, m3, m4, m5 = st.columns(5)

        completed = sum(1 for s in result.trace if s.status == "completed")
        conf = result.synthesis_report.confidence_score if result.synthesis_report else 0
        counter_count = len(result.synthesis_report.counterarguments) if result.synthesis_report else 0
        gap_count = len(result.synthesis_report.evidence_gaps) if result.synthesis_report else 0
        cite_count = len(result.retrieval_result.cases_retrieved) if result.retrieval_result else 0

        conf_color = "confidence-high" if conf >= 70 else ("confidence-medium" if conf >= 40 else "confidence-low")

        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{completed}/6</div><div class="metric-label">Agents Completed</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value {conf_color}">{conf}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cite_count}</div><div class="metric-label">Real Cases Found</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{counter_count}</div><div class="metric-label">Counterarguments</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{gap_count}</div><div class="metric-label">Evidence Gaps</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Row 2: Two columns — Conclusion + Gaps
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### ⚖️ Conclusion")
            if result.synthesis_report:
                sr = result.synthesis_report
                st.markdown(f'<div class="card"><div class="card-header">Executive Summary</div><div class="card-text">{sr.executive_summary}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><div class="card-header">Leaning Conclusion</div><div class="card-text">{sr.leaning_conclusion}</div></div>', unsafe_allow_html=True)

            # Counterarguments
            if result.synthesis_report and result.synthesis_report.counterarguments:
                st.markdown("### ⚡ Counterarguments (Opposing View)")
                for ca in result.synthesis_report.counterarguments:
                    strength = ca.strength if hasattr(ca, 'strength') else "moderate"
                    arg = ca.argument if hasattr(ca, 'argument') else str(ca)
                    legal = ca.legal_basis if hasattr(ca, 'legal_basis') else ""
                    st.markdown(
                        f'<div class="counter-card">'
                        f'<div class="strength">[{strength.upper()}]</div>'
                        f'<div class="argument">{arg}</div>'
                        + (f'<div style="color:#a5b4fc;font-size:0.8rem;margin-top:0.3rem;">{legal}</div>' if legal else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        with col_right:
            # Evidence Gaps
            st.markdown("### ⚠️ Evidence Gaps")
            gaps_found = False
            if result.synthesis_report and result.synthesis_report.evidence_gaps:
                for g in result.synthesis_report.evidence_gaps:
                    desc = g.description if hasattr(g, 'description') else str(g)
                    imp = g.importance if hasattr(g, 'importance') else ""
                    if desc.strip():
                        gaps_found = True
                        st.markdown(
                            f'<div class="gap-warning">'
                            f'<strong>⚠ {desc}</strong>'
                            + (f'<br><small>{imp}</small>' if imp.strip() else "")
                            + '</div>',
                            unsafe_allow_html=True,
                        )
            if not gaps_found:
                st.info("No critical evidence gaps identified. The available facts provide good coverage.")

            # Key Findings
            st.markdown("### 🔑 Key Findings")
            if result.synthesis_report and result.synthesis_report.key_findings:
                for f in result.synthesis_report.key_findings:
                    st.markdown(f'<div class="finding-item">{f}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Row 3: Retrieved Cases
        st.markdown("### 📚 Real Cases Retrieved from Swiss Federal Court")
        if result.retrieval_result and result.retrieval_result.cases_retrieved:
            case_cols = st.columns(min(3, len(result.retrieval_result.cases_retrieved)))
            for i, c in enumerate(result.retrieval_result.cases_retrieved[:6]):
                with case_cols[i % 3]:
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="card-header">📜 {c.citation_string or c.decision_id}</div>'
                        f'<div class="card-text" style="font-size:0.8rem; color:#94a3b8;">{c.court_name} · {c.decision_date}</div>'
                        f'<div class="card-text" style="margin-top:0.5rem;">{(c.regeste or c.rule_statement or "")[:200]}...</div>'
                        + (f'<a href="{c.canonical_url}" target="_blank" style="color:#60a5fa;font-size:0.8rem;">View full decision →</a>' if c.canonical_url else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No cases retrieved. Try with DeepSeek API for better search query generation.")

        st.markdown("---")

        # Row 4: Reasoning Trace (expandable)
        with st.expander("🔍 Full Reasoning Trace (Click to expand)", expanded=False):
            st.caption("Every agent step is logged with input, output, confidence, and timing.")
            for step in result.trace:
                icon = "✅" if step.status == "completed" else "❌"
                st.markdown(
                    f"**{icon} {step.agent_name}** · {step.confidence}% confidence · {step.duration_ms}ms",
                )
                st.caption(step.output_summary[:150])
                if step.error:
                    st.error(step.error)

        # Row 5: Applicable Statutes
        if result.retrieval_result and result.retrieval_result.statutes_retrieved:
            with st.expander("📖 Applicable Swiss Statutes (Code of Obligations)", expanded=False):
                for s in result.retrieval_result.statutes_retrieved[:8]:
                    st.markdown(f'**Art. {s.article_num} OR** — *{s.title[:80] if s.title else "Obligationenrecht"}*')
                    st.markdown(f'<div class="card-text" style="margin-bottom:1rem;">{s.text[:400]}</div>', unsafe_allow_html=True)

        # Disclaimer
        st.markdown("---")
        st.warning(
            "⚠️ **Research Prototype Disclaimer:** This is a decision-support tool for exploring precedents "
            "and reasoning transparency. It is NOT legal advice. All conclusions must be reviewed "
            "by a qualified lawyer. Outputs may contain errors. Designed to demonstrate transparent, "
            "sovereign AI reasoning methods for Swiss Tenancy Law (Mietrecht, Art. 253-274g OR)."
        )

# ── Tab 2: Custom Analysis ──────────────────────────────────────────────────

with tab_manual:
    st.markdown("### 📝 Analyze Your Own Tenancy Dispute")
    st.caption("Paste a description of a Swiss tenancy law dispute. The system will retrieve real cases, analyze the legal issues, and produce a structured reasoning report.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area(
            "Describe the dispute",
            height=200,
            placeholder="Example: I have been renting an apartment in Zurich since 2019. On January 15, 2024, my landlord sent me a termination notice by regular mail without using the official cantonal form. The reason given was 'personal use for his son.' My monthly rent is CHF 1,850. The termination is effective March 31, 2024. I believe the termination is invalid because the required form was not used...",
            key="manual_text",
        )
    with col2:
        st.markdown("**Claim Details**")
        user_role = st.selectbox("Your Role", ["tenant", "landlord"], key="manual_role")
        user_type = st.selectbox(
            "Dispute Type",
            ["termination_validity", "rent_increase", "defect_remediation", "deposit_dispute", "other"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="manual_type",
        )
        user_lang = st.selectbox("Language", ["de", "fr", "it"], key="manual_lang")

    if st.button("🔍 Analyze My Claim", type="primary", use_container_width=True, disabled=not user_text.strip()):
        with st.spinner("Running multi-agent reasoning pipeline..."):
            config = LLMConfig(provider=provider, model=model)
            if api_key:
                config.api_key = api_key
            orchestrator = Orchestrator(llm_config=config)
            try:
                claim = ClaimInput(
                    raw_text=user_text,
                    claimant_role=PartyRole(user_role),
                    claim_type=ClaimType(user_type),
                    canton="CH",
                    language=user_lang,
                )
                result = orchestrator.run(claim)
                st.session_state.manual_result = result
            finally:
                orchestrator.close()

    if "manual_result" in st.session_state:
        result = st.session_state.manual_result
        if result.synthesis_report:
            sr = result.synthesis_report
            st.success("✅ Analysis complete!")
            st.markdown(f"**Confidence:** {sr.confidence_score}%")

            st.markdown("#### Executive Summary")
            st.info(sr.executive_summary)

            st.markdown("#### Leaning Conclusion")
            st.warning(sr.leaning_conclusion)

            if sr.key_findings:
                st.markdown("#### Key Findings")
                for f in sr.key_findings:
                    st.markdown(f"- {f}")

            if sr.counterarguments:
                st.markdown("#### Counterarguments")
                for ca in sr.counterarguments:
                    st.markdown(f"- **[{ca.strength}]** {ca.argument}")
        elif result.error:
            st.error(f"Analysis failed: {result.error}")

# ── Tab 3: About ────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("### ℹ️ About SovereignLex")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("""
        **SovereignLex** is a research prototype demonstrating transparent, explainable multi-agent AI reasoning for legal analysis.

        #### Core Research Question
        > Can a transparent multi-agent reasoning architecture with explicit task decomposition, evidence mapping, verification steps, uncertainty estimation, and counterargument generation improve the traceability and reliability of legal AI while preserving digital sovereignty?

        #### How It Works (The 6-Agent Pipeline)

        ```
        Your Claim
            │
            ▼
        ┌──────────────┐
        │ Coordinator  │  "Let me plan what we need to analyze"
        │   (Planner)  │  → Identifies legal issues, relevant statutes
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │    Fact      │  "Let me extract all the facts and dates"
        │  Extractor   │  → Structured facts, timeline, parties
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │   Legal      │  "Let me search the real Swiss case database"
        │  Retriever   │  → OpenCaseLaw API: 191K+ real court decisions
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  Evidence    │  "Let me map facts against the law"
        │   Mapper     │  → Evidence gaps, supporting/contradicting evidence
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  Verifier    │  "Let me check for mistakes"
        │   (Auditor)  │  → Citation accuracy, logical gaps, confidence adjustment
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Synthesizer  │  "Here's the final report"
        │   (Report)   │  → Summary + confidence + gaps + counterarguments
        └──────────────┘
        ```

        #### Key Features
        - **Real case retrieval** from OpenCaseLaw (not AI hallucinations)
        - **Full reasoning trace** — every step logged and auditable
        - **Evidence gaps** — what critical information is missing
        - **Counterarguments** — built-in devil's advocate
        - **Confidence scoring** with justification
        - **Sovereign** processing with local LLM option (Ollama)
        - **Multi-provider** — DeepSeek, Ollama, or OpenAI

        #### Domain
        Swiss Tenancy Law (Mietrecht, Art. 253–274g OR). The architecture generalizes to any legal domain.

        #### Technology
        Python 3.12 · Pydantic v2 · LiteLLM · Instructor · Streamlit · OpenCaseLaw REST API · Ollama
        """)

    with col_b:
        st.markdown("#### Quick Stats")
        st.metric("Agents in Pipeline", "6")
        st.metric("OpenCaseLaw Cases", "191K+")
        st.metric("Swiss Courts", "121")
        st.metric("Statute Articles", "Art. 253-274g OR")
        st.metric("Example Cases", "5")

        st.markdown("#### Data Sources")
        st.markdown("- [OpenCaseLaw REST API](https://opencaselaw.ch/api/)")
        st.markdown("- [Swiss Code of Obligations](https://www.fedlex.admin.ch/)")
        st.markdown("- [BGE/BGER Federal Court](https://www.bger.ch/)")

        st.markdown("#### Repository")
        st.markdown("[github.com/priyankanagabhushana/multi-agent-legal-ai](https://github.com/priyankanagabhushana/multi-agent-legal-ai)")

    st.markdown("---")
    st.warning(
        "⚠️ **Research Prototype Disclaimer:** This is a decision-support tool for exploring precedents "
        "and reasoning transparency. It is NOT legal advice. All conclusions must be reviewed "
        "by a qualified lawyer. Outputs may contain errors."
    )
