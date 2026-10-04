from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_CROSS_ENCODER_MODELS = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name

    def _load_model(self):
        if self.model_name not in _CROSS_ENCODER_MODELS:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER_MODELS[self.model_name] = CrossEncoder(self.model_name)
        return _CROSS_ENCODER_MODELS[self.model_name]

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [
            (query, doc["text"] if isinstance(doc, dict) else getattr(doc, "text", ""))
            for doc in documents
        ]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            text = doc["text"] if isinstance(doc, dict) else getattr(doc, "text", "")
            orig_score = doc.get("score", 0.0) if isinstance(doc, dict) else getattr(doc, "score", 0.0)
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            results.append(RerankResult(
                text=text,
                original_score=float(orig_score),
                rerank_score=float(score),
                metadata=meta,
                rank=i
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        from flashrank import RerankRequest
        model = self._load_model()
        passages = [
            {"id": i, "text": d["text"] if isinstance(d, dict) else getattr(d, "text", ""), "meta": d}
            for i, d in enumerate(documents)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        results = model.rerank(rerank_req)
        output = []
        for i, res in enumerate(results[:top_k]):
            meta_doc = res.get("meta", {})
            orig_score = meta_doc.get("score", 0.0) if isinstance(meta_doc, dict) else 0.0
            meta = meta_doc.get("metadata", {}) if isinstance(meta_doc, dict) else {}
            output.append(RerankResult(
                text=res["text"],
                original_score=float(orig_score),
                rerank_score=float(res["score"]),
                metadata=meta,
                rank=i
            ))
        return output


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
