# evaluate_herald.py
import os
import json
from datetime import datetime

from research_analyst.orchestration.orchestrator import ResearchAnalyst
from research_analyst.utils.metrics import EvaluationSuite

analyst = ResearchAnalyst()
suite   = EvaluationSuite()

# ── Test queries with ground truths ──────────────────────────────────
queries = [
    {
        "question": "How did Sam Altman's firing and rehiring at OpenAI in November 2023 affect the company's relationships with Microsoft, its investors, and the broader AI industry?",
        "ground_truth": "Sam Altman was fired by OpenAI's board in November 2023 and rehired days later. Microsoft, a major investor and partner, was not informed of the firing. Over 700 of 770 employees threatened to resign. Microsoft briefly hired Altman before he returned as CEO. The event raised serious concerns about AI governance and OpenAI's stability.",
        "reference_facts": [
            "Sam Altman was fired by OpenAI board in November 2023",
            "700 of 770 employees threatened to leave",
            "Microsoft was not informed about the firing",
            "Microsoft hired Altman after OpenAI rejected his return",
            "Altman was eventually rehired as CEO of OpenAI",
        ]
    },
    {
        "question": "What is the relationship between OpenAI and Microsoft?",
        "ground_truth": "Microsoft has invested billions of dollars in OpenAI and is its primary commercial partner. Microsoft integrates OpenAI models into Azure, Bing, and Copilot. The partnership gives Microsoft exclusive cloud computing rights for OpenAI technology.",
        "reference_facts": [
            "Microsoft invested billions in OpenAI",
            "Microsoft integrates OpenAI into Azure",
            "Microsoft Copilot uses OpenAI technology",
        ]
    },
    {
        "question": "What is GraphRAG and how does it differ from standard RAG?",
        "ground_truth": "GraphRAG combines knowledge graphs with retrieval-augmented generation. It extracts entities and relationships to enable multi-hop reasoning. Standard RAG only retrieves documents by semantic similarity. GraphRAG can answer complex relational queries by traversing graph relationships across multiple documents.",
        "reference_facts": [
            "GraphRAG uses knowledge graphs",
            "GraphRAG enables multi-hop reasoning",
            "Standard RAG uses semantic similarity only",
            "GraphRAG extracts entities and relationships",
        ]
    },
    {
        "question": "Who founded OpenAI and when?",
        "ground_truth": "OpenAI was founded in December 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, Elon Musk, Wojciech Zaremba, and John Schulman among others. It was established as a non-profit AI research company.",
        "reference_facts": [
            "OpenAI founded in December 2015",
            "Sam Altman is a co-founder",
            "Elon Musk is a co-founder",
            "Founded as non-profit AI research company",
        ]
    },
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language model generation. The system first retrieves relevant documents from a knowledge base, then uses a language model to generate an answer grounded in the retrieved content. RAG reduces hallucination by grounding answers in retrieved facts.",
        "reference_facts": [
            "RAG combines retrieval with generation",
            "RAG retrieves relevant documents first",
            "RAG reduces hallucination",
            "RAG grounds answers in retrieved content",
        ]
    },
]

# ── Run evaluation ────────────────────────────────────────────────────
results = []

print("=" * 60)
print("HERALD EVALUATION")
print("=" * 60)

for i, q in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] {q['question'][:65]}...")

    response = analyst.query(q["question"])

    if not response.success or not response.answer:
        print(f"  FAILED: {response.error}")
        results.append({
            "question": q["question"],
            "error": response.error,
            "success": False
        })
        continue

    prediction = response.answer.text

    # ── NLP metrics ──────────────────────────────────────────────────
    nlp = suite.evaluate(
        prediction      = prediction,
        reference       = q["ground_truth"],
        reference_facts = q["reference_facts"],
    )

    # ── Internal HERALD metrics ───────────────────────────────────────
    qm = response.quality_metrics
    internal = {}
    if qm:
        internal = {
            "composite_score":    round(qm.composite_score or 0, 4),
            "citation_coverage":  round(qm.citation_coverage,    4),
            "grounding_score":    round(qm.grounding_score,       4),
            "coherence_score":    round(qm.coherence_score,       4),
            "answer_completeness":round(qm.answer_completeness,   4),
            "source_diversity":   round(qm.source_diversity,      4),
        }

    result = {
        "question":       q["question"],
        "success":        True,
        "execution_path": str(response.execution_path),
        "execution_time_ms": round(response.execution_time_ms, 1),
        "answer_length":  len(prediction),
        "num_citations":  len(response.answer.citations),
        "nlp_metrics":    nlp,
        "internal_metrics": internal,
    }
    results.append(result)

    # Print per-query summary
    print(f"  Path:      {response.execution_path}")
    print(f"  Time:      {response.execution_time_ms/1000:.1f}s")
    print(f"  Composite: {internal.get('composite_score', 'N/A')}")
    print(f"  CitCov:    {internal.get('citation_coverage', 'N/A')}")
    print(f"  ROUGE-1:   {nlp.get('rouge1_f', 0):.4f}")
    print(f"  ROUGE-L:   {nlp.get('rougeL_f', 0):.4f}")
    print(f"  METEOR:    {nlp.get('meteor', 0):.4f}")
    print(f"  Sem Sim:   {nlp.get('sem_sim', 0):.4f}")
    print(f"  Fact Cov:  {nlp.get('fact_cov', 0):.4f}")

# ── Aggregate results ─────────────────────────────────────────────────
successful = [r for r in results if r.get("success")]

if successful:
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    def avg(key, sub=None):
        vals = []
        for r in successful:
            d = r[sub] if sub else r
            if key in d and d[key] is not None:
                vals.append(d[key])
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print(f"\n  Samples evaluated:    {len(successful)}/{len(queries)}")
    print(f"  Avg execution time:   {avg('execution_time_ms')/1000:.1f}s")
    print()
    print("  Internal Metrics:")
    print(f"    Composite Score:    {avg('composite_score',    'internal_metrics')}")
    print(f"    Citation Coverage:  {avg('citation_coverage',  'internal_metrics')}")
    print(f"    Grounding Score:    {avg('grounding_score',    'internal_metrics')}")
    print(f"    Coherence Score:    {avg('coherence_score',    'internal_metrics')}")
    print(f"    Answer Completeness:{avg('answer_completeness','internal_metrics')}")
    print(f"    Source Diversity:   {avg('source_diversity',   'internal_metrics')}")
    print()
    print("  NLP Metrics (vs ground truth):")
    print(f"    ROUGE-1 F:          {avg('rouge1_f', 'nlp_metrics')}")
    print(f"    ROUGE-2 F:          {avg('rouge2_f', 'nlp_metrics')}")
    print(f"    ROUGE-L F:          {avg('rougeL_f', 'nlp_metrics')}")
    print(f"    BLEU-4:             {avg('bleu4',    'nlp_metrics')}")
    print(f"    METEOR:             {avg('meteor',   'nlp_metrics')}")
    print(f"    Semantic Sim:       {avg('sem_sim',  'nlp_metrics')}")
    print(f"    Fact Coverage:      {avg('fact_cov', 'nlp_metrics')}")

# ── Save to JSON ──────────────────────────────────────────────────────
out_path = f"results/herald_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Full results saved to: {out_path}")