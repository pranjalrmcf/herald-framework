from research_analyst.orchestration.orchestrator import ResearchAnalyst
import os
analyst = ResearchAnalyst()
response = analyst.query("How did Sam Altman's firing and rehiring at OpenAI in November 2023 affect the company's relationships with Microsoft, its investors, and the broader AI industry?")
# response = analyst.query("what has been mgk's best era?")
if response.success and response.answer:
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.answer.text)
    print()
    print("Execution path:", response.execution_path)
    print("Composite score:", response.quality_metrics.composite_score if response.quality_metrics else "N/A")
    print("Citations:", len(response.answer.citations))
else:
    print("Failed:", response.error)


# uvicorn research_analyst.api.main:app --reload --host 0.0.0.0 --port 8000
# python -m research_analyst.test