"""Core agents for the SovereignLex multi-agent reasoning pipeline.

Each agent:
- Takes structured Pydantic input
- Returns structured Pydantic output
- Includes confidence score (0-100) and reasoning notes
- Logs its work for the reasoning trace
"""

import json
import time
from typing import Optional

from .api_client import OpenCaseLawClient
from .llm import LLM
from .models import (
    AnalysisPlan,
    ClaimInput,
    ClaimType,
    CounterArgument,
    EvidenceGap,
    EvidenceItem,
    EvidenceMapping,
    EvidenceStatus,
    ExtractedFacts,
    PartyRole,
    PipelineResult,
    PlanStep,
    RecommendedAction,
    RetrievalResult,
    RetrievedCase,
    RetrievedStatute,
    SynthesisReport,
    TimelineEvent,
    TraceStep,
    VerificationFinding,
    VerificationReport,
)


# ── CoordinatorAgent ────────────────────────────────────────────────────────


class CoordinatorAgent:
    """Decomposes the user's claim into a structured analysis plan."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, claim: ClaimInput) -> tuple[AnalysisPlan, TraceStep]:
        t0 = time.monotonic()
        prompt = f"""Analyze this Swiss tenancy law claim and produce a structured analysis plan.

CLAIM TEXT:
{claim.raw_text}

CLAIMANT ROLE: {claim.claimant_role.value}
CLAIM TYPE: {claim.claim_type.value}
CANTON: {claim.canton}

Identify:
1. The core legal issues (focus on Swiss Tenancy Law - Mietrecht, OR Art. 253-274g)
2. Key statutes that are likely relevant (e.g., Art. 266l OR for termination forms, Art. 271 OR for abusive termination, Art. 259 OR for defects)
3. A step-by-step plan for analysis (what agents to call in what order)

Return the analysis as a structured plan."""
        plan = self.llm.generate_structured(prompt, AnalysisPlan)
        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="coordinator",
            agent_name="CoordinatorAgent",
            input_summary=f"Claim: {claim.raw_text[:150]}...",
            output_summary=f"Plan: {len(plan.steps)} steps, {len(plan.legal_issues)} issues",
            confidence=plan.confidence,
            duration_ms=duration,
        )
        return plan, trace


# ── FactExtractorAgent ──────────────────────────────────────────────────────


class FactExtractorAgent:
    """Extracts structured facts, dates, parties, and timeline from the claim."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, claim: ClaimInput, plan: AnalysisPlan) -> tuple[ExtractedFacts, TraceStep]:
        t0 = time.monotonic()
        prompt = f"""Extract structured facts from this Swiss tenancy law claim.

CLAIM TEXT:
{claim.raw_text}

CLAIMANT ROLE: {claim.claimant_role.value}
ATTACHED DOCUMENTS: {claim.attached_documents}

Extract:
- Parties (names and roles — tenant, landlord, subtenant)
- Property address
- Monthly rent (if mentioned)
- Lease start date, termination date
- Termination reason given
- Key communications between parties
- Any claimed defects
- A chronological timeline of events with dates

Be precise. If something is NOT mentioned, leave it empty or null. Do not invent facts.
Flag absences of critical information in extraction_notes."""
        facts = self.llm.generate_structured(prompt, ExtractedFacts)
        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="fact_extraction",
            agent_name="FactExtractorAgent",
            input_summary=f"Claim text ({len(claim.raw_text)} chars)",
            output_summary=f"Facts: {len(facts.timeline)} events, parties={list(facts.parties.keys())}",
            confidence=facts.confidence,
            duration_ms=duration,
        )
        return facts, trace


# ── LegalRetrieverAgent ─────────────────────────────────────────────────────


class LegalRetrieverAgent:
    """Retrieves relevant Swiss cases and statutes from OpenCaseLaw."""

    TENANCY_STATUTES = [
        "253", "253a", "253b", "254", "255", "256", "256a", "256b",
        "257", "257a", "257b", "257c", "257d", "257e", "257f",
        "258", "259", "259a", "259b", "259c", "259d", "259e", "259f", "259g",
        "260", "261", "261a", "261b", "262", "263", "264",
        "265", "266", "266a", "266b", "266c", "266d", "266e", "266f",
        "266g", "266h", "266i", "266k", "266l", "266m", "266n", "266o",
        "267", "267a", "267b", "268", "268a", "268b", "269", "269a", "269b",
        "269c", "269d", "269e", "270", "270a", "270b", "270c", "270d", "270e",
        "271", "271a", "272", "272a", "272b", "272c", "272d",
        "273", "273a", "273b", "273c", "274", "274a", "274b", "274c",
        "274d", "274e", "274f", "274g",
    ]

    def __init__(self, api: OpenCaseLawClient, llm: Optional[LLM] = None):
        self.api = api
        self.llm = llm

    def _build_search_queries(self, facts: ExtractedFacts, plan: AnalysisPlan) -> list[str]:
        """Build targeted search queries based on extracted facts and legal issues."""
        queries = []

        base = "Mietrecht"
        if plan.legal_issues:
            queries.append(f"{base} {' '.join(plan.legal_issues[:3])}")

        if facts.termination_reason:
            queries.append(f"{base} Kündigung {' '.join(facts.termination_reason.split()[:5])}")

        if facts.defects_claimed:
            queries.append(f"{base} Mängel {' '.join(facts.defects_claimed[0].split()[:5])}")

        for statute in plan.key_statutes[:5]:
            article = statute.replace("Art. ", "").replace(" OR", "").strip()
            queries.append(f"{base} Art. {article}")

        if not queries:
            queries.append(f"{base} Kündigung Anfechtung")

        return queries[:5]

    def _get_relevant_statutes(self, plan: AnalysisPlan) -> list[RetrievedStatute]:
        """Fetch statute text for key articles identified in the plan."""
        statutes = []
        seen = set()

        for statute_ref in plan.key_statutes[:8]:
            try:
                article = statute_ref.replace("Art. ", "").replace(" OR", "").strip()
                if article in seen:
                    continue
                seen.add(article)
                data = self.api.get_law_article("OR", article)
                for art in data.get("articles", []):
                    statutes.append(RetrievedStatute(
                        abbreviation="OR",
                        article_num=art["article_num"],
                        title=data.get("title", ""),
                        text=art.get("text", ""),
                        relevance=f"Relevant to: {statute_ref}",
                        consolidation_date=data.get("consolidation_date", ""),
                    ))
            except Exception:
                continue

        if not statutes:
            for article in ["271", "266l", "257d", "259"]:
                if article in seen:
                    continue
                seen.add(article)
                try:
                    data = self.api.get_law_article("OR", article)
                    for art in data.get("articles", []):
                        statutes.append(RetrievedStatute(
                            abbreviation="OR",
                            article_num=art["article_num"],
                            title=data.get("title", ""),
                            text=art.get("text", ""),
                            relevance="Core tenancy statute",
                            consolidation_date=data.get("consolidation_date", ""),
                        ))
                except Exception:
                    continue

        return statutes

    def run(
        self, claim: ClaimInput, facts: ExtractedFacts, plan: AnalysisPlan
    ) -> tuple[RetrievalResult, TraceStep]:
        t0 = time.monotonic()
        queries = self._build_search_queries(facts, plan)
        all_cases: list[RetrievedCase] = []
        total_found = 0

        for query in queries[:3]:
            for court_filter in ["bge", "bger"]:
                try:
                    result = self.api.search_decisions(
                        q=query,
                        limit=5,
                        language=claim.language,
                        court=court_filter,
                        canton=claim.canton if claim.canton != "CH" else None,
                    )
                    total_found += result.get("total", 0)
                    for item in result.get("results", []):
                        all_cases.append(RetrievedCase(
                            decision_id=item.get("decision_id", ""),
                            court=item.get("court", ""),
                            court_name=item.get("court_name", ""),
                            canton=item.get("canton", ""),
                            chamber=item.get("chamber", ""),
                            docket_number=item.get("docket_number", ""),
                            decision_date=item.get("decision_date", ""),
                            language=item.get("language", ""),
                            title=item.get("title"),
                            regeste=item.get("regeste", ""),
                            rule_statement=item.get("rule_statement", ""),
                            citation_string=item.get("citation_string_de", ""),
                            canonical_url=item.get("canonical_url", ""),
                            relevance_score=item.get("relevance_score", 0.0),
                            statutes_cited=item.get("statutes", []),
                            is_leading_case=item.get("is_leading_case", False),
                            citation_count=item.get("citation_count", 0),
                        ))
                except Exception:
                    continue

        seen_ids = set()
        unique_cases = []
        for c in all_cases:
            if c.decision_id not in seen_ids:
                seen_ids.add(c.decision_id)
                unique_cases.append(c)

        unique_cases.sort(key=lambda c: c.relevance_score, reverse=True)

        statutes = self._get_relevant_statutes(plan)

        result = RetrievalResult(
            query_used=queries[0] if queries else "",
            statutes_retrieved=statutes,
            cases_retrieved=unique_cases[:10],
            leading_cases=[c for c in unique_cases if c.is_leading_case],
            total_cases_found=total_found,
            total_statutes_found=len(statutes),
            search_notes=f"Searched {len(queries)} queries, found {total_found} total cases, kept {len(unique_cases)} unique.",
            confidence=min(90, 50 + len(unique_cases) * 5),
        )
        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="legal_retrieval",
            agent_name="LegalRetrieverAgent",
            input_summary=f"Queries: {queries[:3]}",
            output_summary=f"Retrieved: {len(unique_cases)} cases, {len(statutes)} statutes",
            confidence=result.confidence,
            duration_ms=duration,
        )
        return result, trace


# ── EvidenceMappingAgent ────────────────────────────────────────────────────


class EvidenceMappingAgent:
    """Maps extracted facts against retrieved cases and statutes."""

    def __init__(self, llm: LLM):
        self.llm = llm

    @staticmethod
    def _generate_deterministic_gaps(facts: ExtractedFacts) -> list[EvidenceGap]:
        """Generate evidence gaps based on what critical information is missing from extracted facts."""
        gaps = []

        if not facts.property_address or facts.property_address == "None":
            gaps.append(EvidenceGap(
                description="Property address not identified",
                importance="Critical for establishing jurisdiction and identifying the rental property",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=15,
            ))

        if not facts.termination_date:
            gaps.append(EvidenceGap(
                description="No termination date specified or extracted",
                importance="Essential for determining notice period compliance and deadline for contestation",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=20,
            ))

        if not facts.monthly_rent:
            gaps.append(EvidenceGap(
                description="Monthly rent amount not specified",
                importance="Needed to verify rent level, calculate deposit, and assess proportionality of claims",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=10,
            ))

        if not facts.lease_start_date:
            gaps.append(EvidenceGap(
                description="Lease start date not found",
                importance="Determines duration of tenancy, which affects notice periods and tenant protection",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=10,
            ))

        if not facts.termination_reason:
            gaps.append(EvidenceGap(
                description="Termination reason not documented or extracted",
                importance="Needed to assess whether termination is abusive (Art. 271 OR) or has valid grounds",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=15,
            ))

        if not facts.key_communications:
            gaps.append(EvidenceGap(
                description="No written communications between parties found",
                importance="Written correspondence (emails, letters) is critical evidence for establishing timeline and notice",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=15,
            ))

        if not facts.timeline:
            gaps.append(EvidenceGap(
                description="No chronological timeline of events could be constructed",
                importance="Timeline is essential for verifying notice periods and sequence of events",
                status=EvidenceStatus.MISSING,
                impact_on_confidence=15,
            ))

        return gaps

    def run(
        self,
        claim: ClaimInput,
        facts: ExtractedFacts,
        retrieval: RetrievalResult,
    ) -> tuple[EvidenceMapping, TraceStep]:
        t0 = time.monotonic()

        case_summaries = []
        for c in retrieval.cases_retrieved[:5]:
            case_summaries.append(
                f"- {c.citation_string}: {c.regeste[:300] if c.regeste else 'No regeste'}"
            )

        statute_summaries = []
        for s in retrieval.statutes_retrieved[:5]:
            statute_summaries.append(f"- Art. {s.article_num} OR: {s.text[:200]}")

        prompt = f"""Map the evidence in this Swiss tenancy law claim against relevant statutes and cases.

CLAIM: {claim.raw_text[:500]}
EXTRACTED FACTS: {json.dumps(facts.model_dump(), indent=2, default=str)[:1000]}

RELEVANT STATUTES:
{chr(10).join(statute_summaries)}

RELEVANT CASES:
{chr(10).join(case_summaries)}

Identify:
1. Claim elements (what needs to be proven)
2. Supporting evidence for each element
3. Contradicting evidence
4. Evidence gaps — what critical information is missing
5. How retrieved cases align with or differ from current facts

Return structured mapping."""
        mapping = self.llm.generate_structured(prompt, EvidenceMapping)

        det_gaps = self._generate_deterministic_gaps(facts)
        llm_gap_descriptions = {g.description.lower() for g in mapping.gaps if g.description}
        for dg in det_gaps:
            if dg.description.lower() not in llm_gap_descriptions:
                mapping.gaps.append(dg)

        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="evidence_mapping",
            agent_name="EvidenceMappingAgent",
            input_summary=f"Mapping facts against {len(retrieval.cases_retrieved)} cases",
            output_summary=f"Elements: {len(mapping.claim_elements)}, gaps: {len(mapping.gaps)}",
            confidence=mapping.confidence,
            duration_ms=duration,
        )
        return mapping, trace


# ── VerifierAgent ───────────────────────────────────────────────────────────


class VerifierAgent:
    """Reviews reasoning for consistency, citation accuracy, logical gaps, hallucinations."""

    def __init__(self, llm: LLM, api: Optional[OpenCaseLawClient] = None):
        self.llm = llm
        self.api = api

    def run(
        self,
        claim: ClaimInput,
        facts: ExtractedFacts,
        retrieval: RetrievalResult,
        mapping: EvidenceMapping,
    ) -> tuple[VerificationReport, TraceStep]:
        t0 = time.monotonic()

        case_citations = [c.citation_string for c in retrieval.cases_retrieved if c.citation_string]
        gaps_text = "\n".join(f"- {g.description}" for g in mapping.gaps)
        evidence_text = "\n".join(
            f"- {e.claim_element}: {e.direction}" for e in mapping.supporting_evidence[:10]
        )

        prompt = f"""Verify the reasoning quality of this Swiss tenancy law analysis.

CLAIM: {claim.raw_text[:400]}
FACTS EXTRACTED: {json.dumps(facts.model_dump(), indent=2, default=str)[:800]}
RETRIEVED CASES: {case_citations[:5]}
EVIDENCE GAPS: {gaps_text[:500]}
EVIDENCE MAPPING: {evidence_text[:500]}

Check for:
1. Logical gaps in the reasoning chain
2. Citation accuracy concerns
3. Missing critical evidence
4. Overconfidence or underconfidence
5. Hallucination risks

Provide a structured verification report with confidence adjustment (-50 to +50)."""
        report = self.llm.generate_structured(prompt, VerificationReport)
        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="verification",
            agent_name="VerifierAgent",
            input_summary=f"Verifying reasoning with {len(retrieval.cases_retrieved)} cases",
            output_summary=f"Findings: {len(report.findings)}, adjustment: {report.confidence_adjustment:+d}",
            confidence=max(0, min(100, 70 + report.confidence_adjustment)),
            duration_ms=duration,
        )
        return report, trace


# ── SynthesizerAgent (includes Counterargument generation) ──────────────────


class SynthesizerAgent:
    """Generates the final structured report with counterarguments, confidence, and actions."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(
        self,
        claim: ClaimInput,
        plan: AnalysisPlan,
        facts: ExtractedFacts,
        retrieval: RetrievalResult,
        mapping: EvidenceMapping,
        verification: VerificationReport,
    ) -> tuple[SynthesisReport, TraceStep]:
        t0 = time.monotonic()

        case_text = "\n".join(
            f"- {c.citation_string} ({c.decision_date}): {c.regeste[:200] if c.regeste else ''}"
            for c in retrieval.cases_retrieved[:5]
        )
        statute_text = "\n".join(
            f"- Art. {s.article_num} OR: {s.text[:200]}" for s in retrieval.statutes_retrieved[:5]
        )
        gaps_text = "\n".join(f"- {g.description} (importance: {g.importance})" for g in mapping.gaps)
        findings_text = "\n".join(f"- [{f.severity}] {f.description}" for f in verification.findings[:5])

        prompt = f"""Generate the final structured legal analysis report for this Swiss tenancy law case.

CLAIM: {claim.raw_text[:500]}
CLAIMANT: {claim.claimant_role.value}
LEGAL ISSUES: {plan.legal_issues}
EXTRACTED FACTS: {json.dumps(facts.model_dump(), indent=2, default=str)[:800]}
APPLICABLE STATUTES:
{statute_text}

KEY PRECEDENTS:
{case_text}

EVIDENCE GAPS FROM MAPPING:
{gaps_text}

VERIFICATION FINDINGS:
{findings_text}

Generate a complete, substantive report:

1. **Executive Summary** (2-3 sentences summarizing the case and key findings)

2. **Claim Analysis** (analyze the legal claim in the context of Swiss tenancy law)

3. **Key Findings** (specific, concrete findings. Each should reference a statute or precedent where applicable)

4. **Key Precedents** — List the most important cases with their citation and explain how each relates to the current claim. Include URLs from the provided data.

5. **Applicable Statutes** — List statutes with article number, text, and how they apply to this case.

6. **Evidence Gaps** — For EACH gap, describe:
   - WHAT specific document or fact is missing
   - WHY it matters for the legal analysis
   - HOW it affects confidence
   Examples: "No signed copy of the official termination form (Art. 266l OR)", "No payment receipts to verify rent amount", "No photos of claimed defects"

7. **Counterarguments** — Generate 2-3 STRONG counterarguments from the opposing party's perspective. Each should:
   - State the argument clearly
   - Reference relevant statutes or precedents
   - Suggest a possible rebuttal
   The counterarguments must be realistic and legally grounded, not straw men.

8. **Confidence Score** (0-100) with detailed justification explaining WHY this confidence level. Be honest about uncertainty.

9. **Leaning Conclusion** — What does the system lean toward based on the analysis?

10. **Recommended Actions** — Concrete next steps for the claimant, with priority level (high/medium/low).

CRITICAL: Evidence gaps and counterarguments MUST be specific and substantive — not generic placeholders. Each should demonstrate legal reasoning.
Be honest about uncertainty. Include the standard legal disclaimer."""
        report = self.llm.generate_structured(prompt, SynthesisReport)
        duration = int((time.monotonic() - t0) * 1000)

        trace = TraceStep(
            step_id="synthesis",
            agent_name="SynthesizerAgent",
            input_summary="All pipeline outputs",
            output_summary=f"Report: {report.confidence_score}% confidence, {len(report.counterarguments)} counterarguments, {len(report.evidence_gaps)} gaps",
            confidence=report.confidence_score,
            duration_ms=duration,
        )
        return report, trace
