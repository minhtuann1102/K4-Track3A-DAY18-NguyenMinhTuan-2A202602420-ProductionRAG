from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, GEMINI_API_KEY, GEMINI_BASE_URL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(val: any) -> float:
    try:
        import math
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _tokenize_text(text: str) -> set[str]:
    stopwords = {
        "là", "và", "của", "có", "trong", "được", "cho", "với", "về", "các",
        "những", "khi", "để", "thì", "ra", "ở", "này", "đó", "một", "theo",
        "không", "phải", "đã", "bị", "sẽ", "như", "nào", "gì", "ai", "bao",
        "nhiêu", "the", "a", "an", "is", "of", "and", "to", "in"
    }
    tokens = set(re.findall(r"\w+", text.lower()))
    filtered = {t for t in tokens if len(t) > 1 and t not in stopwords}
    return filtered if filtered else tokens


def _compute_offline_metrics(questions: list[str], answers: list[str],
                             contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question: list[EvalResult] = []

    for q, a, ctx_list, gt in zip(questions, answers, contexts, ground_truths):
        q_tokens = _tokenize_text(q)
        a_tokens = _tokenize_text(a)
        gt_tokens = _tokenize_text(gt)
        all_ctx_text = " ".join(ctx_list)
        all_ctx_tokens = _tokenize_text(all_ctx_text)

        # 1. Faithfulness: Is answer grounded in context?
        if not a.strip() or a == "Không tìm thấy thông tin." or not ctx_list:
            faith = 0.0
        else:
            supported = len(a_tokens & all_ctx_tokens)
            ratio = supported / max(len(a_tokens), 1)
            faith = round(min(1.0, 0.4 + 0.6 * ratio), 4)

        # 2. Answer Relevancy: Does answer address the question?
        if not a.strip() or a == "Không tìm thấy thông tin.":
            rel = 0.0
        else:
            q_match = len(a_tokens & q_tokens) / max(len(q_tokens), 1)
            gt_match = len(a_tokens & gt_tokens) / max(len(gt_tokens), 1)
            rel = round(min(1.0, 0.35 + 0.35 * q_match + 0.30 * gt_match), 4)

        # 3. Context Recall: Are ground truth facts covered in retrieved contexts?
        if not gt_tokens or not all_ctx_tokens:
            recall = 0.0
        else:
            gt_covered = len(gt_tokens & all_ctx_tokens)
            recall = round(min(1.0, gt_covered / len(gt_tokens)), 4)

        # 4. Context Precision: Are relevant contexts ranked at the top?
        if not ctx_list or not gt_tokens:
            prec = 0.0
        else:
            precisions = []
            rel_count = 0
            for rank_idx, ctx_text in enumerate(ctx_list):
                c_tokens = _tokenize_text(ctx_text)
                overlap = len(gt_tokens & c_tokens) / len(gt_tokens)
                if overlap >= 0.2:
                    rel_count += 1
                    precisions.append(rel_count / (rank_idx + 1))
            prec = round(sum(precisions) / max(len(precisions), 1), 4) if precisions else 0.0

        per_question.append(
            EvalResult(
                question=q,
                answer=a,
                contexts=ctx_list,
                ground_truth=gt,
                faithfulness=faith,
                answer_relevancy=rel,
                context_precision=prec,
                context_recall=recall,
            )
        )

    n = max(len(per_question), 1)
    return {
        "faithfulness": round(sum(r.faithfulness for r in per_question) / n, 4),
        "answer_relevancy": round(sum(r.answer_relevancy for r in per_question) / n, 4),
        "context_precision": round(sum(r.context_precision for r in per_question) / n, 4),
        "context_recall": round(sum(r.context_recall for r in per_question) / n, 4),
        "per_question": per_question,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation with offline fallback if API key is not configured."""
    has_gemini = bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AIzaSy..."))
    has_openai = bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."))
    if has_gemini or has_openai:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            kwargs = {}
            if has_gemini:
                try:
                    from langchain_openai import ChatOpenAI
                    kwargs["llm"] = ChatOpenAI(
                        api_key=GEMINI_API_KEY,
                        base_url=GEMINI_BASE_URL,
                        model="gemini-1.5-flash",
                        temperature=0,
                    )
                except Exception:
                    pass

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                **kwargs
            )
            df = result.to_pandas()
            per_question = [
                EvalResult(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=list(row["contexts"]),
                    ground_truth=str(row["ground_truth"]),
                    faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                    answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                    context_precision=_safe_float(row.get("context_precision", 0.0)),
                    context_recall=_safe_float(row.get("context_recall", 0.0)),
                )
                for _, row in df.iterrows()
            ]
            return {
                "faithfulness": _safe_float(result.get("faithfulness", 0.0)),
                "answer_relevancy": _safe_float(result.get("answer_relevancy", 0.0)),
                "context_precision": _safe_float(result.get("context_precision", 0.0)),
                "context_recall": _safe_float(result.get("context_recall", 0.0)),
                "per_question": per_question,
            }
        except Exception as e:
            print(f"  ⚠️  RAGAS online evaluation failed: {e}. Switching to offline evaluation...")

    return _compute_offline_metrics(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_items = []
    for res in eval_results:
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall,
        }
        avg_score = sum(metrics.values()) / max(len(metrics), 1)
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown failure", "Review query and context")
        )

        scored_items.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
