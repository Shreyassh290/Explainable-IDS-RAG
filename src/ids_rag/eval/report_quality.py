"""Report quality: BERTScore, ROUGE and BLEU against per-class reference texts, with paired RAG vs vanilla tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon

from ids_rag.config import get_project_root

CLASSES = ("Benign", "DoS", "DDoS")


def clean(text: str) -> str:
    """Remove markdown symbols and [n] citation markers so both modes are scored on plain text."""
    text = re.sub(r"\[\d{1,2}\]", "", text)
    text = re.sub(r"[#*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment of p-values."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def boot_ci(diff: list[float], n: int = 10000, seed: int = 42) -> tuple[float, float]:
    """95 percent bootstrap interval of the mean paired difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="artifacts/eval/flow_summary.csv")
    ap.add_argument("--flows-dir", default="artifacts/eval/flows")
    ap.add_argument("--ref-dir", default="data/eval")
    ap.add_argument("--bert-model", default="distilroberta-base")
    ap.add_argument("--bert-layers", type=int, default=5)
    ap.add_argument("--no-bert", action="store_true", help="skip BERTScore (no model download)")
    ap.add_argument("--raw", action="store_true", help="score the report text without cleaning")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = get_project_root()
    prep = (lambda s: s) if args.raw else clean
    ref_paths = {c: root / args.ref_dir / f"reference_{c.lower()}.txt" for c in CLASSES}
    refs = {c: prep(p.read_text(encoding="utf-8-sig")) for c, p in ref_paths.items()}
    ref_words = {c: len(t.split()) for c, t in refs.items()}
    rows = list(csv.DictReader(open(root / args.summary, encoding="utf-8")))
    recs = []
    for r in rows:
        path = root / args.flows_dir / f"flow_{r['index']}_{r['mode']}.txt"
        recs.append({"index": int(r["index"]), "mode": r["mode"], "true_class": r["true_class"],
                     "cand": prep(path.read_text(encoding="utf-8-sig"))})
    cands = [x["cand"] for x in recs]
    ref_list = [refs[x["true_class"]] for x in recs]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for x, ref in zip(recs, ref_list):
        s = scorer.score(ref, x["cand"])
        x["rouge1"], x["rouge2"], x["rougeL"] = (s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL"))
        x["bleu"] = sacrebleu.sentence_bleu(x["cand"], [ref]).score / 100.0
        x["words_scored"] = len(x["cand"].split())
    metrics = ["rouge1", "rouge2", "rougeL", "bleu"]
    if not args.no_bert:
        from bert_score import BERTScorer
        bs = BERTScorer(model_type=args.bert_model, num_layers=args.bert_layers, lang="en", rescale_with_baseline=False)
        p_, r_, f_ = bs.score(cands, ref_list)
        for x, p, r, f in zip(recs, p_.tolist(), r_.tolist(), f_.tolist()):
            x["bertscore_p"], x["bertscore_r"], x["bertscore_f1"] = p, r, f
        metrics = ["bertscore_f1", "bertscore_p", "bertscore_r"] + metrics
    out_dir = root / "artifacts" / "eval"
    with open(out_dir / "report_quality_per_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_class", "mode", "words_scored"] + metrics)
        for x in sorted(recs, key=lambda z: (z["index"], z["mode"])):
            w.writerow([x["index"], x["true_class"], x["mode"], x["words_scored"]] + [round(x[m], 6) for m in metrics])
    by = {(x["index"], x["mode"]): x for x in recs}
    idx = sorted({x["index"] for x in recs if (x["index"], "rag") in by and (x["index"], "vanilla") in by})
    print(f"paired flows: {len(idx)} | cleaning: {'none' if args.raw else 'markdown and [n] removed'} | bertscore: {'skipped' if args.no_bert else args.bert_model + ' layer ' + str(args.bert_layers)}")
    print("reference words after cleaning:", ref_words, "| mean words scored: rag", round(st.mean(by[(i, 'rag')]['words_scored'] for i in idx), 1), "vanilla", round(st.mean(by[(i, 'vanilla')]['words_scored'] for i in idx), 1))
    pvals, table = {}, {}
    for m in metrics:
        a = np.array([by[(i, "rag")][m] for i in idx])
        b = np.array([by[(i, "vanilla")][m] for i in idx])
        d = a - b
        try:
            p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        pvals[m] = p
        lo, hi = boot_ci(d.tolist())
        table[m] = {"vanilla": float(b.mean()), "rag": float(a.mean()), "diff": float(d.mean()), "median_diff": float(np.median(d)),
                    "rel_pct": float(100 * d.mean() / b.mean()) if b.mean() else float("nan"),
                    "wins": int((d > 1e-9).sum()), "ties": int((abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum()),
                    "p": p, "ci_lo": lo, "ci_hi": hi}
    adj = holm(pvals)
    print(f"{'metric':14s} {'vanilla':>8s} {'rag':>8s} {'diff':>8s} {'rel%':>8s}  W/T/L   {'p':>9s} {'p_holm':>9s}  95% CI of mean diff")
    for m in metrics:
        t = table[m]
        t["p_holm"] = adj[m]
        print(f"{m:14s} {t['vanilla']:8.4f} {t['rag']:8.4f} {t['diff']:+8.4f} {t['rel_pct']:+8.1f}  {t['wins']}/{t['ties']}/{t['losses']}  {t['p']:9.2e} {t['p_holm']:9.2e}  [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]")
    corp = {}
    for mode in ("vanilla", "rag"):
        sub = [by[(i, mode)] for i in idx]
        corp[mode] = sacrebleu.corpus_bleu([x["cand"] for x in sub], [[refs[x["true_class"]] for x in sub]]).score / 100.0
    print(f"corpus BLEU: vanilla {corp['vanilla']:.4f} | rag {corp['rag']:.4f}")
    print("per-class means (vanilla -> rag):")
    for c in CLASSES:
        ci = [i for i in idx if by[(i, "rag")]["true_class"] == c]
        if ci:
            print(f"  {c:7s} n={len(ci):2d} " + " | ".join(f"{m} {st.mean(by[(i,'vanilla')][m] for i in ci):.3f}->{st.mean(by[(i,'rag')][m] for i in ci):.3f}" for m in metrics))
    json.dump({"paired_flows": len(idx), "cleaning": "none" if args.raw else "markdown and [n] removed",
               "bert_model": None if args.no_bert else args.bert_model, "bert_layers": None if args.no_bert else args.bert_layers,
               "reference_sha256": {c: sha(p) for c, p in ref_paths.items()}, "reference_words": ref_words,
               "corpus_bleu": corp, "metrics": table}, open(out_dir / "report_quality_summary.json", "w", encoding="utf-8"), indent=2)
    print("wrote", out_dir / "report_quality_per_report.csv", "and report_quality_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
