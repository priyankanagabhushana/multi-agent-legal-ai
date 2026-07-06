"""Orchestrator — runs the multi-agent reasoning pipeline with full trace logging.

The orchestrator:
1. Takes a ClaimInput
2. Runs agents sequentially: Coordinator → FactExtractor → Retriever → EvidenceMapping → Verifier → Synthesizer
3. Logs every step with timestamps, inputs, outputs, confidence
4. Handles errors gracefully with partial results
5. Returns a complete PipelineResult
"""

import time
from typing import Optional

from .agents import (
    CoordinatorAgent,
    EvidenceMappingAgent,
    FactExtractorAgent,
    LegalRetrieverAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from .api_client import OpenCaseLawClient
from .llm import LLM, LLMConfig
from .models import (
    ClaimInput,
    PipelineResult,
    TraceStep,
)


class Orchestrator:
    """Central pipeline orchestrator for SovereignLex."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        api: Optional[OpenCaseLawClient] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        if llm is not None:
            self.llm = llm
        elif llm_config is not None:
            self.llm = LLM(llm_config)
        else:
            self.llm = LLM()
        self.api = api or OpenCaseLawClient()

        self.coordinator = CoordinatorAgent(self.llm)
        self.fact_extractor = FactExtractorAgent(self.llm)
        self.retriever = LegalRetrieverAgent(self.api, self.llm)
        self.evidence_mapper = EvidenceMappingAgent(self.llm)
        self.verifier = VerifierAgent(self.llm, self.api)
        self.synthesizer = SynthesizerAgent(self.llm)

    def run(self, claim: ClaimInput) -> PipelineResult:
        """Execute the full reasoning pipeline and return structured results."""
        t_start = time.monotonic()
        trace: list[TraceStep] = []
        result = PipelineResult(claim_input=claim)

        try:
            plan, t = self.coordinator.run(claim)
            result.analysis_plan = plan
            trace.append(t)
        except Exception as e:
            trace.append(TraceStep(
                step_id="coordinator",
                agent_name="CoordinatorAgent",
                status="failed",
                error=str(e),
            ))
            result.status = "failed"
            result.error = f"Coordinator failed: {e}"
            result.trace = trace
            return result

        try:
            facts, t = self.fact_extractor.run(claim, plan)
            result.extracted_facts = facts
            trace.append(t)
        except Exception as e:
            trace.append(TraceStep(
                step_id="fact_extraction",
                agent_name="FactExtractorAgent",
                status="failed",
                error=str(e),
            ))
            result.status = "partial"
            result.error = f"FactExtractor failed: {e}"

        if result.extracted_facts:
            try:
                retrieval, t = self.retriever.run(claim, result.extracted_facts, plan)
                result.retrieval_result = retrieval
                trace.append(t)
            except Exception as e:
                trace.append(TraceStep(
                    step_id="legal_retrieval",
                    agent_name="LegalRetrieverAgent",
                    status="failed",
                    error=str(e),
                ))
                if result.status != "failed":
                    result.status = "partial"

        if result.extracted_facts and result.retrieval_result:
            try:
                mapping, t = self.evidence_mapper.run(
                    claim, result.extracted_facts, result.retrieval_result
                )
                result.evidence_mapping = mapping
                trace.append(t)
            except Exception as e:
                trace.append(TraceStep(
                    step_id="evidence_mapping",
                    agent_name="EvidenceMappingAgent",
                    status="failed",
                    error=str(e),
                ))

        if result.extracted_facts and result.retrieval_result and result.evidence_mapping:
            try:
                verification, t = self.verifier.run(
                    claim,
                    result.extracted_facts,
                    result.retrieval_result,
                    result.evidence_mapping,
                )
                result.verification_report = verification
                trace.append(t)
            except Exception as e:
                trace.append(TraceStep(
                    step_id="verification",
                    agent_name="VerifierAgent",
                    status="failed",
                    error=str(e),
                ))

        if (
            result.analysis_plan
            and result.extracted_facts
            and result.retrieval_result
            and result.evidence_mapping
            and result.verification_report
        ):
            try:
                report, t = self.synthesizer.run(
                    claim,
                    result.analysis_plan,
                    result.extracted_facts,
                    result.retrieval_result,
                    result.evidence_mapping,
                    result.verification_report,
                )
                result.synthesis_report = report
                trace.append(t)

                if (not report.evidence_gaps or all(
                    not g.description or not g.description.strip() for g in report.evidence_gaps
                )) and result.evidence_mapping and result.evidence_mapping.gaps:
                    report.evidence_gaps = [
                        g for g in result.evidence_mapping.gaps if g.description and g.description.strip()
                    ]
            except Exception as e:
                trace.append(TraceStep(
                    step_id="synthesis",
                    agent_name="SynthesizerAgent",
                    status="failed",
                    error=str(e),
                ))

        result.total_duration_ms = int((time.monotonic() - t_start) * 1000)
        result.trace = trace

        if result.status == "failed":
            pass
        elif result.synthesis_report:
            result.status = "completed"
        elif result.error:
            result.status = "partial"
        else:
            result.status = "completed"

        return result

    def get_trace_summary(self, result: PipelineResult) -> str:
        """Generate a human-readable trace summary."""
        lines = []
        for step in result.trace:
            status_icon = "✓" if step.status == "completed" else "✗"
            lines.append(
                f"  {status_icon} [{step.agent_name}] "
                f"confidence={step.confidence}% "
                f"({step.duration_ms}ms)"
            )
            if step.output_summary:
                lines.append(f"    → {step.output_summary}")
            if step.error:
                lines.append(f"    ⚠ {step.error}")
        lines.append(f"\n  Total: {result.total_duration_ms}ms | Status: {result.status}")
        return "\n".join(lines)

    def close(self):
        self.api.close()
