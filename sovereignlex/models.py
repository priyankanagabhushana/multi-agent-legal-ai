"""Pydantic models for SovereignLex — all agent I/O contracts.

Every agent receives and returns structured JSON via these models.
This enables testing, replay, debugging, and audit trails.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Helpers ─────────────────────────────────────────────────────────────────


def _normalize_string_list(v: Any) -> list[str]:
    """Normalize LLM output: lists of objects/dicts → list of strings."""
    if not isinstance(v, list):
        return []
    result = []
    for item in v:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            for val in item.values():
                if isinstance(val, str):
                    result.append(val)
                    break
            else:
                result.append(str(item))
        else:
            result.append(str(item))
    return result


def _normalize_dict_values(v: Any) -> dict:
    """Ensure dict values are strings, converting nested dicts to their string representation."""
    if not isinstance(v, dict):
        return {}
    return {k: str(val) if not isinstance(val, str) else val for k, val in v.items()}


def _normalize_nulls(data: dict, string_fields: list[str]) -> dict:
    """Replace None/null values with empty strings for specified fields."""
    for field in string_fields:
        if data.get(field) is None:
            data[field] = ""
    return data


# ── Enums ────────────────────────────────────────────────────────────────────


class PartyRole(str, Enum):
    TENANT = "tenant"
    LANDLORD = "landlord"
    SUBTENANT = "subtenant"
    OTHER = "other"


class ClaimType(str, Enum):
    TERMINATION_VALIDITY = "termination_validity"
    RENT_INCREASE = "rent_increase"
    DEFECT_REMEDIATION = "defect_remediation"
    DEPOSIT_DISPUTE = "deposit_dispute"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    PARTIAL = "partial"
    CONTRADICTORY = "contradictory"


# ── Input Models ────────────────────────────────────────────────────────────


class ClaimInput(BaseModel):
    """Raw user claim — the entry point to the pipeline."""

    raw_text: str = Field(description="Full text of the claim or uploaded document")
    claimant_role: PartyRole = Field(
        default=PartyRole.TENANT, description="Who is bringing the claim"
    )
    claim_type: ClaimType = Field(
        default=ClaimType.TERMINATION_VALIDITY,
        description="Primary type of legal dispute",
    )
    canton: str = Field(default="CH", description="Two-letter canton code or CH")
    language: str = Field(default="de", description="Document language (de/fr/it)")
    attached_documents: list[str] = Field(
        default_factory=list,
        description="Names/descriptions of attached supporting documents",
    )


# ── Coordinator / Planner Output ────────────────────────────────────────────


class PlanStep(BaseModel):
    """A single step in the analysis plan."""

    step_id: str = Field(default="", description="Unique step identifier")
    agent: str = Field(default="", description="Agent responsible for this step")
    description: str = Field(default="", description="What this step should accomplish")
    key_questions: list[str] = Field(
        default_factory=list, description="Specific questions to answer"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Step IDs this step depends on"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "questions" in data and not data.get("key_questions"):
                data["key_questions"] = data["questions"]
            if "step" in data and not data.get("step_id"):
                data["step_id"] = str(data["step"])
        return data


class AnalysisPlan(BaseModel):
    """Coordinator output — the decomposition of the legal analysis."""

    claim_summary: str = Field(default="", description="One-paragraph summary of the claim")
    legal_issues: list[str] = Field(default_factory=list, description="Identified legal issues")
    key_statutes: list[str] = Field(
        default_factory=list,
        description="Potentially relevant statute articles (e.g., Art. 271 OR)",
    )
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered plan steps")
    confidence: int = Field(default=50, ge=0, le=100, description="Planner confidence in its decomposition (0-100)")
    reasoning_notes: str = Field(default="", description="Planner's reasoning")

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["legal_issues"] = _normalize_string_list(data.get("legal_issues", []))
            data["key_statutes"] = _normalize_string_list(data.get("key_statutes", []))
            data = _normalize_nulls(data, ["claim_summary", "reasoning_notes"])
        return data


# ── Fact Extraction Output ──────────────────────────────────────────────────


class TimelineEvent(BaseModel):
    """A single event in the case timeline."""

    date: str = Field(default="", description="Date of the event (ISO format or approximate)")
    description: str = Field(default="", description="What happened")
    source: str = Field(default="claim", description="Where this fact came from")
    confidence: int = Field(default=100, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "event" in data and not data.get("description"):
                data["description"] = data["event"]
        return data


class ExtractedFacts(BaseModel):
    """FactExtractorAgent output — structured facts from the claim."""

    parties: dict = Field(default_factory=dict, description="Party names and roles")
    property_address: Any = Field(default="", description="Rental property address")
    monthly_rent: Any = Field(default=None)
    lease_start_date: Any = Field(default=None)
    termination_date: Any = Field(default=None)
    termination_reason: Any = Field(default=None)
    key_communications: list[str] = Field(
        default_factory=list,
        description="Key communications between parties (emails, letters)",
    )
    defects_claimed: list[str] = Field(
        default_factory=list, description="Claimed defects in the property"
    )
    timeline: list[TimelineEvent] = Field(
        default_factory=list, description="Chronological timeline of events"
    )
    confidence: int = Field(default=50, ge=0, le=100, description="Overall confidence in fact extraction")
    extraction_notes: str = Field(default="", description="Notes about extraction challenges or ambiguities")

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["key_communications"] = _normalize_string_list(data.get("key_communications", []))
            data["defects_claimed"] = _normalize_string_list(data.get("defects_claimed", []))
            data["parties"] = _normalize_dict_values(data.get("parties", {}))
            for field in ["property_address", "monthly_rent", "lease_start_date",
                          "termination_date", "termination_reason", "extraction_notes"]:
                if data.get(field) is None:
                    data[field] = ""
        return data


# ── Legal Retrieval Output ──────────────────────────────────────────────────


class RetrievedStatute(BaseModel):
    """A retrieved statute article."""

    abbreviation: str = Field(default="OR", description="e.g., OR, ZGB")
    article_num: str = Field(default="", description="e.g., 271, 266l")
    title: str = Field(default="")
    text: str = Field(default="", description="Full text of the article")
    relevance: str = Field(default="", description="Why this statute is relevant")
    consolidation_date: str = Field(default="")


class RetrievedCase(BaseModel):
    """A retrieved legal case."""

    decision_id: str = Field(default="", description="OpenCaseLaw decision ID")
    court: str = Field(default="", description="Court abbreviation")
    court_name: str = Field(default="")
    canton: str = Field(default="CH")
    chamber: str = Field(default="")
    docket_number: str = Field(default="")
    decision_date: str = Field(default="")
    language: str = Field(default="de")
    title: Optional[str] = Field(default=None)
    regeste: str = Field(default="", description="Official headnote/summary")
    rule_statement: str = Field(default="")
    citation_string: str = Field(default="", description="e.g., BGE 140 III 86")
    canonical_url: str = Field(default="")
    relevance_score: float = Field(default=0.0)
    relevance_explanation: str = Field(default="", description="Why this case is relevant to the claim")
    statutes_cited: list[str] = Field(default_factory=list)
    is_leading_case: bool = Field(default=False)
    citation_count: int = Field(default=0)
    full_text_snippet: str = Field(default="")


class RetrievalResult(BaseModel):
    """LegalRetrieverAgent output."""

    query_used: str = Field(default="", description="Search query sent to OpenCaseLaw")
    statutes_retrieved: list[RetrievedStatute] = Field(default_factory=list)
    cases_retrieved: list[RetrievedCase] = Field(default_factory=list)
    leading_cases: list[RetrievedCase] = Field(default_factory=list)
    total_cases_found: int = Field(default=0)
    total_statutes_found: int = Field(default=0)
    search_notes: str = Field(default="")
    confidence: int = Field(default=50, ge=0, le=100)


# ── Evidence Mapping Output ─────────────────────────────────────────────────


class EvidenceGap(BaseModel):
    """A missing or weak piece of evidence."""

    description: str = Field(default="", description="What evidence is missing")
    importance: str = Field(default="", description="Why this evidence matters")
    status: EvidenceStatus = Field(default=EvidenceStatus.MISSING)
    impact_on_confidence: int = Field(default=0, ge=0, le=100, description="How much this gap reduces confidence")


class EvidenceItem(BaseModel):
    """A piece of evidence mapped to a claim element."""

    claim_element: str = Field(default="", description="What part of the claim this supports")
    evidence_description: str = Field(default="", description="The evidence")
    source: str = Field(default="", description="Where this comes from")
    direction: str = Field(default="supporting", description="supporting, contradicting, or neutral")
    confidence: int = Field(default=50, ge=0, le=100)


class EvidenceMapping(BaseModel):
    """EvidenceMappingAgent output — maps facts to legal elements."""

    claim_elements: list[str] = Field(default_factory=list, description="Individual elements of the claim to be proven")
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    case_similarities: list[dict] = Field(default_factory=list, description="How retrieved cases align with or differ from current facts")
    confidence: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["claim_elements"] = _normalize_string_list(data.get("claim_elements", []))
            data = _normalize_nulls(data, [])
        return data


# ── Verification Output ─────────────────────────────────────────────────────


class VerificationFinding(BaseModel):
    """A single finding from the verification process."""

    severity: str = Field(default="info", description="critical, warning, info")
    category: str = Field(default="", description="e.g., citation_error, logical_gap, missing_evidence")
    description: str = Field(default="", description="What was found")
    recommendation: str = Field(default="", description="How to fix or address it")


class VerificationReport(BaseModel):
    """VerifierAgent output."""

    findings: list[VerificationFinding] = Field(default_factory=list)
    logical_gaps: list[str] = Field(default_factory=list, description="Identified gaps in reasoning")
    missing_evidence_list: list[str] = Field(default_factory=list, description="Key evidence that is missing")
    confidence_adjustment: int = Field(default=0, description="Adjustment to overall confidence (-50 to +50)")
    overall_verdict: str = Field(default="", description="One-paragraph assessment of reasoning quality")
    additional_retrieval_needed: bool = Field(default=False)
    additional_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["logical_gaps"] = _normalize_string_list(data.get("logical_gaps", []))
            data["missing_evidence_list"] = _normalize_string_list(data.get("missing_evidence_list", []))
            data["additional_queries"] = _normalize_string_list(data.get("additional_queries", []))
            data = _normalize_nulls(data, ["overall_verdict"])
        return data


# ── Counterargument Output ──────────────────────────────────────────────────


class CounterArgument(BaseModel):
    """A single counterargument from the devil's advocate."""

    argument: str = Field(default="", description="The counterargument")
    legal_basis: str = Field(default="", description="Statutes or precedents supporting this argument")
    strength: str = Field(default="moderate", description="How strong this counterargument is")
    rebuttal: str = Field(default="", description="Possible rebuttal to this counterargument")


# ── Final Synthesis / Report ────────────────────────────────────────────────


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of the final confidence score."""

    fact_quality: int = Field(default=50, ge=0, le=100)
    legal_basis_strength: int = Field(default=50, ge=0, le=100)
    precedent_alignment: int = Field(default=50, ge=0, le=100)
    evidence_completeness: int = Field(default=50, ge=0, le=100)
    counterargument_resilience: int = Field(default=50, ge=0, le=100)


class RecommendedAction(BaseModel):
    action: str = Field(default="")
    priority: str = Field(default="medium")
    rationale: str = Field(default="")


class SynthesisReport(BaseModel):
    """SynthesizerAgent output — the final structured report."""

    executive_summary: str = Field(default="", description="One-paragraph summary")
    claim_analysis: str = Field(default="", description="Analysis of the claim")
    key_findings: list[str] = Field(default_factory=list)
    key_precedents: list[dict] = Field(default_factory=list, description="Most relevant cases with links and relevance explanation")
    applicable_statutes: list[dict] = Field(default_factory=list, description="Relevant statutes with text and application")
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    counterarguments: list[CounterArgument] = Field(default_factory=list)
    confidence_score: int = Field(default=50, ge=0, le=100, description="Overall confidence 0-100")
    confidence_breakdown: Optional[ConfidenceBreakdown] = Field(default=None)
    confidence_justification: str = Field(default="", description="Why this confidence score")
    leaning_conclusion: str = Field(default="", description="The system's leaning on the claim")
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    disclaimer: str = Field(
        default="This is a research prototype for decision support. "
        "It is NOT legal advice. All conclusions must be reviewed "
        "by a qualified lawyer. Outputs may contain errors."
    )
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["key_findings"] = _normalize_string_list(data.get("key_findings", []))
        return data


# ── Trace / Pipeline Models ─────────────────────────────────────────────────


class TraceStep(BaseModel):
    """A single logged step in the pipeline execution."""

    step_id: str = Field(default="")
    agent_name: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_summary: str = Field(default="")
    output_summary: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)
    duration_ms: int = Field(default=0)
    status: str = Field(default="completed")
    error: Optional[str] = Field(default=None)


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""

    claim_input: Optional[ClaimInput] = Field(default=None)
    analysis_plan: Optional[AnalysisPlan] = Field(default=None)
    extracted_facts: Optional[ExtractedFacts] = Field(default=None)
    retrieval_result: Optional[RetrievalResult] = Field(default=None)
    evidence_mapping: Optional[EvidenceMapping] = Field(default=None)
    verification_report: Optional[VerificationReport] = Field(default=None)
    synthesis_report: Optional[SynthesisReport] = Field(default=None)
    trace: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: int = Field(default=0)
    status: str = Field(default="completed")
    error: Optional[str] = Field(default=None)
