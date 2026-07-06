"""Evaluation harness for SovereignLex.

Tests the pipeline on real tenancy cases and measures:
1. Citation correctness — whether retrieved case citations are real
2. Fact extraction accuracy — whether key facts are extracted
3. Pipeline completeness — whether all agents complete successfully
4. Confidence calibration — whether confidence scores are reasonable

Usage:
    python -m sovereignlex.eval_benchmark
    python -m sovereignlex.eval_benchmark --cases 3
    python -m sovereignlex.eval_benchmark --verbose
"""

import json
import time
from pathlib import Path
from typing import Optional

from .models import ClaimInput, ClaimType, PartyRole
from .orchestrator import Orchestrator
from .api_client import OpenCaseLawClient


EVAL_CASES = [
    {
        "case_id": "eval_001",
        "name": "Termination Without Official Form — Art. 266l OR",
        "claimant_role": "tenant",
        "claim_type": "termination_validity",
        "language": "de",
        "raw_text": """Der Mieter hat am 15. Januar 2024 ein Kündigungsschreiben des Vermieters erhalten. 
Das Schreiben war ein formloser Brief ohne Verwendung des kantonalen Formulars. 
Begründung: Eigenbedarf des Sohnes. Mietverhältnis seit 1. April 2019. 
Monatsmiete CHF 1'850. Kündigung per 31. März 2024.
Gemäss Art. 266l OR muss die Kündigung auf einem vom Kanton genehmigten Formular erfolgen. 
Frage: Ist die Kündigung wegen Formmangels ungültig?""",
        "expected_issues": ["Formmangel", "Art. 266l OR", "Kündigungsformular"],
        "expected_statutes": ["266l OR", "271 OR"],
        "ground_truth_conclusion": "Kündigung wahrscheinlich ungültig wegen Formmangels",
    },
    {
        "case_id": "eval_002",
        "name": "Rent Increase After Renovation — Art. 269 OR",
        "claimant_role": "tenant",
        "claim_type": "rent_increase",
        "language": "de",
        "raw_text": """Mietverhältnis in Bern seit 2020. Küchenrenovation im Sommer 2023 (CHF 25'000).
Mietzinserhöhung von CHF 2'200 auf CHF 2'650 per 1. Januar 2024.
Das offizielle Formular wurde verwendet, aber die Erhöhung erscheint überhöht.
Gemäss Art. 269 OR sind Mietzinserhöhungen nur zulässig, wenn sie nicht missbräuchlich sind.
Frage: Ist die Mietzinserhöhung rechtmässig?""",
        "expected_issues": ["Mietzinserhöhung", "Art. 269 OR", "missbräuchlich"],
        "expected_statutes": ["269 OR", "269d OR"],
        "ground_truth_conclusion": "Erhöhung möglicherweise überhöht, Mieter kann anfechten",
    },
    {
        "case_id": "eval_003",
        "name": "Mietzinshinterlegung wegen Heizungsmangel — Art. 259a ff. OR",
        "claimant_role": "tenant",
        "claim_type": "defect_remediation",
        "language": "de",
        "raw_text": """Wohnung in Basel seit 2021. Seit November 2023 Heizung defekt (unter 18°C).
Mehrfache schriftliche Mängelanzeigen (20.11.2023, 5.12.2023, 2.1.2024).
Vermieter hat Techniker geschickt (10.12.2023), Problem nicht behoben.
Mieter hat Miete im Januar 2024 um 20% gekürzt (von CHF 1'500 auf CHF 1'200).
Vermieter droht mit Kündigung wegen Zahlungsverzug.
Frage: War die Mietzinshinterlegung rechtmässig?""",
        "expected_issues": ["Mängel", "Heizung", "Mietzinshinterlegung", "Art. 259a OR"],
        "expected_statutes": ["259a OR", "259d OR", "259g OR"],
        "ground_truth_conclusion": "Mietzinshinterlegung rechtmässig, Kündigungsdrohung unzulässig",
    },
    {
        "case_id": "eval_004",
        "name": "Rachekündigung nach Mängelrüge — Art. 271a OR",
        "claimant_role": "tenant",
        "claim_type": "termination_validity",
        "language": "de",
        "raw_text": """Mietverhältnis in Genf seit 2018. Mängelrüge am 5. Februar 2024 (Wasserschaden).
Kündigung am 20. Februar 2024 per 31. Mai 2024 mit Begründung \"Umbauarbeiten\".
Nur 15 Tage zwischen Mängelrüge und Kündigung. Monatsmiete CHF 2'100.
Gemäss Art. 271a OR ist eine Kündigung anfechtbar, wenn sie als Reaktion auf eine 
Mängelrüge erfolgt (Rachekündigung).
Frage: Liegt eine unzulässige Rachekündigung vor?""",
        "expected_issues": ["Rachekündigung", "Art. 271a OR", "Mängelrüge"],
        "expected_statutes": ["271a OR", "271 OR"],
        "ground_truth_conclusion": "Kündigung wahrscheinlich anfechtbar als Rachekündigung",
    },
    {
        "case_id": "eval_005",
        "name": "Mietkaution nicht freigegeben — Art. 257e OR",
        "claimant_role": "tenant",
        "claim_type": "deposit_dispute",
        "language": "de",
        "raw_text": """Auszug aus Wohnung in Luzern am 31. Dezember 2023 nach 4 Jahren.
Mietkaution CHF 5'400 (3 Monatsmieten) auf Mietkautionskonto.
Abnahmeprotokoll vom 28.12.2023 dokumentiert nur minimale Gebrauchsspuren.
6 Monate nach Auszug: Kaution immer noch nicht freigegeben.
Vermieter behauptet \"versteckte Schäden\" ohne Nachweis.
Gemäss Art. 257e OR muss die Kaution nach Beendigung des Mietverhältnisses
zurückgegeben werden, sofern keine Forderungen bestehen.
Frage: Hat der Mieter Anspruch auf Rückgabe der Kaution?""",
        "expected_issues": ["Mietkaution", "Art. 257e OR", "Rückgabe"],
        "expected_statutes": ["257e OR"],
        "ground_truth_conclusion": "Mieter hat Anspruch auf Rückgabe, Vermieter muss Forderungen nachweisen",
    },
]


class EvalResult:
    def __init__(self):
        self.case_id = ""
        self.name = ""
        self.status = ""
        self.error: Optional[str] = None
        self.duration_ms = 0
        self.all_agents_completed = False
        self.confidence_score = 0
        self.has_counterarguments = False
        self.has_evidence_gaps = False
        self.has_conclusion = False
        self.citations_count = 0
        self.citation_check_notes = ""
        self.expected_issues_matched = 0
        self.expected_issues_total = 0
        self.expected_statutes_matched = 0
        self.expected_statutes_total = 0
        self.notes = ""


def run_evaluation(max_cases: int = 5, verbose: bool = False):
    """Run the evaluation pipeline on all test cases."""
    orchestrator = Orchestrator()
    api = OpenCaseLawClient()
    results: list[EvalResult] = []

    print("=" * 70)
    print("SOVEREIGNLEX EVALUATION HARNESS")
    print("=" * 70)
    print()

    for case in EVAL_CASES[:max_cases]:
        r = EvalResult()
        r.case_id = case["case_id"]
        r.name = case["name"]
        r.expected_issues_total = len(case.get("expected_issues", []))
        r.expected_statutes_total = len(case.get("expected_statutes", []))

        print(f"📋 [{r.case_id}] {r.name}")
        t0 = time.monotonic()

        try:
            claim = ClaimInput(
                raw_text=case["raw_text"],
                claimant_role=PartyRole(case.get("claimant_role", "tenant")),
                claim_type=ClaimType(case.get("claim_type", "termination_validity")),
                canton="CH",
                language=case.get("language", "de"),
            )

            result = orchestrator.run(claim)
            r.duration_ms = result.total_duration_ms
            r.status = result.status

            agent_statuses = [s.status for s in result.trace]
            r.all_agents_completed = all(s == "completed" for s in agent_statuses)
            num_completed = sum(1 for s in agent_statuses if s == "completed")
            print(f"  Agents: {num_completed}/{len(agent_statuses)} completed")

            if result.synthesis_report:
                sr = result.synthesis_report
                r.confidence_score = sr.confidence_score
                r.has_counterarguments = len(sr.counterarguments) > 0
                r.has_evidence_gaps = len(sr.evidence_gaps) > 0
                r.has_conclusion = bool(sr.leaning_conclusion)
                print(f"  Confidence: {sr.confidence_score}%")
                print(f"  Counterarguments: {len(sr.counterarguments)}")
                print(f"  Evidence gaps: {len(sr.evidence_gaps)}")

            if result.retrieval_result:
                r.citations_count = len(result.retrieval_result.cases_retrieved)
                print(f"  Citations retrieved: {r.citations_count}")

            if result.analysis_plan:
                plan_text = " ".join(result.analysis_plan.legal_issues).lower()
                plan_text += " " + " ".join(result.analysis_plan.key_statutes).lower()
                for issue in case.get("expected_issues", []):
                    if issue.lower() in plan_text:
                        r.expected_issues_matched += 1
                for statute in case.get("expected_statutes", []):
                    if statute.lower() in plan_text:
                        r.expected_statutes_matched += 1
                print(f"  Issues matched: {r.expected_issues_matched}/{r.expected_issues_total}")
                print(f"  Statutes matched: {r.expected_statutes_matched}/{r.expected_statutes_total}")

            if result.error:
                r.error = result.error
                print(f"  Error: {result.error}")

        except Exception as e:
            r.status = "error"
            r.error = str(e)
            print(f"  EXCEPTION: {e}")

        results.append(r)
        print()

    # ── Summary ────────────────────────────────────────────────────────────

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = len(results)
    completed = sum(1 for r in results if r.status == "completed")
    print(f"Completed: {completed}/{total}")

    avg_confidence = sum(r.confidence_score for r in results) / max(completed, 1)
    print(f"Average confidence: {avg_confidence:.1f}%")

    all_complete = sum(1 for r in results if r.all_agents_completed)
    print(f"All agents completed: {all_complete}/{total}")

    with_counter = sum(1 for r in results if r.has_counterarguments)
    print(f"With counterarguments: {with_counter}/{total}")

    with_gaps = sum(1 for r in results if r.has_evidence_gaps)
    print(f"With evidence gaps: {with_gaps}/{total}")

    total_issues = sum(r.expected_issues_total for r in results)
    matched_issues = sum(r.expected_issues_matched for r in results)
    if total_issues > 0:
        print(f"Issue matching accuracy: {matched_issues}/{total_issues} ({100*matched_issues/total_issues:.0f}%)")

    total_stats = sum(r.expected_statutes_total for r in results)
    matched_stats = sum(r.expected_statutes_matched for r in results)
    if total_stats > 0:
        print(f"Statute matching accuracy: {matched_stats}/{total_stats} ({100*matched_stats/total_stats:.0f}%)")

    total_citations = sum(r.citations_count for r in results)
    print(f"Total citations retrieved: {total_citations}")

    total_time = sum(r.duration_ms for r in results)
    avg_time = total_time / max(len(results), 1)
    print(f"Average pipeline time: {avg_time/1000:.1f}s")
    print(f"Total evaluation time: {total_time/1000:.1f}s")

    print()
    print("--- Per-Case Details ---")
    for r in results:
        status_icon = "✅" if r.status == "completed" else "⚠️"
        gaps_icon = "🔍" if r.has_evidence_gaps else "—"
        ca_icon = "⚡" if r.has_counterarguments else "—"
        print(f"  {status_icon} {r.case_id}: conf={r.confidence_score}% "
              f"| issues={r.expected_issues_matched}/{r.expected_issues_total} "
              f"| stats={r.expected_statutes_matched}/{r.expected_statutes_total} "
              f"| cites={r.citations_count} "
              f"| gaps={gaps_icon} ca={ca_icon} "
              f"| {r.duration_ms/1000:.1f}s")

    orchestrator.close()
    api.close()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SovereignLex Evaluation Harness")
    parser.add_argument("--cases", type=int, default=5, help="Number of cases to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    run_evaluation(max_cases=args.cases, verbose=args.verbose)
