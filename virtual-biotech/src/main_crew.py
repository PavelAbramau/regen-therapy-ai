from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool

try:
    from crewai import LLM
except ImportError:  # pragma: no cover - fallback for older CrewAI versions
    LLM = None  # type: ignore[assignment]


def build_llm():
    """
    Build an LLM configuration for CrewAI.

    Env vars:
    - VBI_LLM_PROVIDER: "openai" (default) or "anthropic"
    - VBI_LLM_MODEL: optional explicit model name

    API keys are read by the underlying provider:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    """
    provider = os.getenv("VBI_LLM_PROVIDER", "openai").strip().lower()
    explicit_model = os.getenv("VBI_LLM_MODEL", "").strip()

    if provider == "anthropic":
        default_model = "anthropic/claude-3-5-sonnet-20241022"
    else:
        default_model = "openai/gpt-4o-mini"

    model = explicit_model or default_model

    # Newer CrewAI versions expose LLM class. If unavailable, pass string model.
    if LLM is not None:
        return LLM(model=model)
    return model


def main() -> None:
    llm = build_llm()

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "strategy.md"
    context_dir = Path(__file__).resolve().parents[1] / "data" / "context"
    docs_tool = DirectoryReadTool(directory=str(context_dir))

    literature_scientist = Agent(
        role="Literature_Scientist",
        goal=(
            "Scour academic databases and summarize epigenetic mechanisms that drive "
            "wound healing and tissue regeneration."
        ),
        backstory=(
            "You are a translational epigenetics researcher with deep expertise in "
            "wound-healing biology, chromatin regulation, and critical literature review."
        ),
        llm=llm,
        verbose=True,
    )

    ip_lawyer = Agent(
        role="IP_Lawyer",
        goal=(
            "Cross-reference scientific findings with patentability requirements and "
            "identify novelty, non-obviousness, and freedom-to-operate risks."
        ),
        backstory=(
            "You are a biotech patent attorney experienced in therapeutic claims, "
            "prior-art analysis, and commercialization strategy."
        ),
        llm=llm,
        tools=[docs_tool],
        verbose=True,
    )

    cso = Agent(
        role="CSO",
        goal=(
            "Synthesize scientific and IP analysis into an actionable strategic report "
            "for a virtual biotech program."
        ),
        backstory=(
            "You are a Chief Scientific Officer who integrates R&D evidence with legal "
            "constraints to prioritize investable, defensible opportunities."
        ),
        llm=llm,
        tools=[docs_tool],
        verbose=True,
    )

    def lawyer_thinking(_):
        print("\n[Thinking] IP_Lawyer is now analyzing patentability...\n")

    def cso_thinking(_):
        print("\n[Thinking] CSO is now synthesizing the final strategy...\n")

    scientist_task = Task(
        description=(
            "Identify and summarize key epigenetic wound-healing mechanisms from the "
            "literature (histone modifications, DNA methylation, ncRNA regulation, "
            "chromatin remodeling). Include high-confidence targets and intervention "
            "hypotheses with concise evidence statements."
        ),
        expected_output=(
            "A structured brief with prioritized epigenetic mechanisms, candidate "
            "targets, and supporting rationale."
        ),
        agent=literature_scientist,
        callback=lawyer_thinking,
    )

    lawyer_task = Task(
        description=(
            "Using the scientist's findings, evaluate each proposed mechanism/target "
            "for patentability. Flag novelty, potential obviousness concerns, broad "
            "claim opportunities, and freedom-to-operate considerations."
        ),
        expected_output=(
            "A patentability matrix with risk level, opportunity notes, and recommended "
            "claim direction per target/mechanism."
        ),
        agent=ip_lawyer,
        context=[scientist_task],
        callback=cso_thinking,
    )

    cso_task = Task(
        description=(
            "Combine scientific and IP outputs into a strategic recommendation for the "
            "next 6-12 months. Provide program priorities, rationale, key risks, and "
            "an execution roadmap suitable for leadership review."
        ),
        expected_output=(
            "A markdown-ready strategy document with executive summary, ranked "
            "opportunities, risk register, and next steps."
        ),
        agent=cso,
        context=[scientist_task, lawyer_task],
    )

    crew = Crew(
        agents=[literature_scientist, ip_lawyer, cso],
        tasks=[scientist_task, lawyer_task, cso_task],
        process=Process.sequential,
        verbose=True,
    )

    print("\n[Thinking] Literature_Scientist is now reviewing literature...\n")
    final_output = crew.kickoff()

    report_content = str(final_output)
    report_path.write_text(report_content, encoding="utf-8")

    print(f"Final strategy report saved to: {report_path}")


if __name__ == "__main__":
    main()
