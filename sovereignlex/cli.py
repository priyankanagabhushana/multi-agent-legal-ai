#!/usr/bin/env python3
"""SovereignLex CLI — command-line interface for the reasoning workbench.

Usage:
    python -m sovereignlex.cli "Claim text here..."
    python -m sovereignlex.cli --example 0
    python -m sovereignlex.cli --file document.txt
    python -m sovereignlex.cli --example 0 --verbose
    python -m sovereignlex.cli --example 0 --json-output result.json
"""

import argparse
import json
import sys
from pathlib import Path

from .models import ClaimInput, ClaimType, PartyRole
from .orchestrator import Orchestrator
from .llm import LLMConfig


def console_display(result, verbose: bool = False):
    """Pretty-print pipeline results to the console."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold blue]SovereignLex[/bold blue] — Explainable Multi-Agent Legal Reasoning Workbench",
        border_style="blue",
    ))

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        return

    console.print("\n[bold]━━━ Reasoning Trace ━━━[/bold]")
    trace_table = Table(show_header=True, header_style="bold")
    trace_table.add_column("Step", style="dim")
    trace_table.add_column("Agent")
    trace_table.add_column("Confidence")
    trace_table.add_column("Duration")
    trace_table.add_column("Output")

    for step in result.trace:
        status = "✓" if step.status == "completed" else "[red]✗[/red]"
        trace_table.add_row(
            status,
            step.agent_name,
            f"{step.confidence}%" if step.confidence else "-",
            f"{step.duration_ms}ms",
            step.output_summary[:100] if step.output_summary else "-",
        )

    console.print(trace_table)

    if verbose and result.analysis_plan:
        console.print("\n[bold]━━━ Analysis Plan ━━━[/bold]")
        console.print(f"[bold]Summary:[/bold] {result.analysis_plan.claim_summary}")
        console.print(f"[bold]Legal Issues:[/bold] {', '.join(result.analysis_plan.legal_issues)}")
        console.print(f"[bold]Key Statutes:[/bold] {', '.join(result.analysis_plan.key_statutes)}")

    if verbose and result.extracted_facts:
        console.print("\n[bold]━━━ Extracted Facts ━━━[/bold]")
        console.print(f"[bold]Parties:[/bold] {result.extracted_facts.parties}")
        console.print(f"[bold]Property:[/bold] {result.extracted_facts.property_address}")
        console.print(f"[bold]Timeline:[/bold] {len(result.extracted_facts.timeline)} events")

    if verbose and result.retrieval_result:
        console.print("\n[bold]━━━ Retrieved Cases ━━━[/bold]")
        for c in result.retrieval_result.cases_retrieved[:5]:
            console.print(f"  • [bold]{c.citation_string}[/bold] ({c.decision_date})")
            if c.regeste:
                console.print(f"    {c.regeste[:200]}...")

    if result.synthesis_report:
        report = result.synthesis_report
        console.print("\n[bold]━━━ Final Report ━━━[/bold]")

        console.print(f"\n[bold yellow]Confidence: {report.confidence_score}%[/bold yellow]")
        console.print(f"[dim]{report.confidence_justification[:200]}[/dim]")

        console.print(f"\n[bold]Executive Summary:[/bold]")
        console.print(report.executive_summary[:500])

        if report.key_findings:
            console.print(f"\n[bold]Key Findings:[/bold]")
            for f in report.key_findings[:5]:
                console.print(f"  • {f}")

        if report.counterarguments:
            console.print(f"\n[bold red]Counterarguments (Opposing View):[/bold red]")
            for ca in report.counterarguments[:3]:
                console.print(f"  ⚡ [bold]{ca.strength.upper()}[/bold]: {ca.argument[:200]}")

        if report.evidence_gaps:
            console.print(f"\n[bold yellow]Evidence Gaps:[/bold yellow]")
            for g in report.evidence_gaps[:5]:
                console.print(f"  ⚠ {g.description}")

        if report.leaning_conclusion:
            console.print(f"\n[bold]Leaning Conclusion:[/bold] {report.leaning_conclusion[:300]}")

        console.print(f"\n[dim italic]{report.disclaimer}[/dim italic]")

    console.print(f"\n[dim]Total pipeline duration: {result.total_duration_ms}ms | Status: {result.status}[/dim]")


def run_cli():
    parser = argparse.ArgumentParser(
        description="SovereignLex — Explainable Multi-Agent Legal Reasoning Workbench"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("text", nargs="?", help="Claim text to analyze")
    group.add_argument("--example", "-e", type=int, help="Run example case by index (0-4)")
    group.add_argument("--list", "-l", action="store_true", help="List available example cases")
    group.add_argument("--file", "-f", type=str, help="Read claim from file")

    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed intermediate results")
    parser.add_argument("--json-output", "-o", type=str, help="Save full JSON output to file")
    parser.add_argument("--role", type=str, default="tenant", choices=["tenant", "landlord"], help="Claimant role")
    parser.add_argument("--type", type=str, default="termination_validity",
                        choices=["termination_validity", "rent_increase", "defect_remediation", "deposit_dispute", "other"],
                        help="Claim type")
    parser.add_argument("--canton", type=str, default="CH", help="Canton code (CH, ZH, BE, etc.)")
    parser.add_argument("--language", type=str, default="de", help="Language code")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["deepseek", "ollama", "openai"],
                        help="LLM provider (default: auto-detect from env)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (e.g., deepseek/deepseek-chat, llama3.1:8b)")

    args = parser.parse_args()

    if args.list:
        from examples.tenancy_cases import EXAMPLES
        for i, ex in enumerate(EXAMPLES):
            print(f"[{i}] {ex['name']}")
            print(f"    {ex['raw_text'][:120]}...")
            print()
        return

    llm_config = None
    if args.provider:
        llm_config = LLMConfig(provider=args.provider, model=args.model)
    elif args.model:
        llm_config = LLMConfig(provider="deepseek", model=args.model)
    orchestrator = Orchestrator(llm_config=llm_config) if llm_config else Orchestrator()

    try:
        if args.example is not None:
            from examples.tenancy_cases import EXAMPLES
            if 0 <= args.example < len(EXAMPLES):
                ex = EXAMPLES[args.example]
                print(f"Running example: {ex['name']}")
                claim = ClaimInput(
                    raw_text=ex["raw_text"],
                    claimant_role=PartyRole(ex.get("claimant_role", "tenant")),
                    claim_type=ClaimType(ex.get("claim_type", "termination_validity")),
                    canton=ex.get("canton", "CH"),
                    language=ex.get("language", "de"),
                    attached_documents=ex.get("attached_documents", []),
                )
            else:
                print(f"Invalid example index. Use 0-{len(EXAMPLES)-1}")
                return
        elif args.file:
            text = Path(args.file).read_text()
            claim = ClaimInput(
                raw_text=text,
                claimant_role=PartyRole(args.role),
                claim_type=ClaimType(args.type),
                canton=args.canton,
                language=args.language,
            )
        else:
            claim = ClaimInput(
                raw_text=args.text,
                claimant_role=PartyRole(args.role),
                claim_type=ClaimType(args.type),
                canton=args.canton,
                language=args.language,
            )

        result = orchestrator.run(claim)
        console_display(result, verbose=args.verbose)

        if args.json_output:
            Path(args.json_output).write_text(
                json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str)
            )
            print(f"\nFull result saved to {args.json_output}")

    finally:
        orchestrator.close()


if __name__ == "__main__":
    run_cli()
