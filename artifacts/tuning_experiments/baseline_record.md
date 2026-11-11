# Baseline Record: Original RAG & Vanilla 30-Flow Evaluation

Source: `artifacts/eval/aggregate.txt`, `artifacts/eval/audit_results_v2.txt`, `artifacts/eval/report_quality_summary.json`
Preserved from original evaluation run on 2026-11-09 / 2026-11-10.

## 1. Automated Aggregate Metrics (30 Flows, 60 Reports)

### RAG Mode (30 reports)
- Words min/mean/max: 214 / 353.5 / 415
- In 300–600 words: 29/30 (97%)
- In 400–500 words: 4/30 (13%)
- All 5 numbered headings: 25/30 (83%)
- All 5 headings in any form: 30/30 (100%)
- At least 1 inline citation: 26/30 (87%)
- Mean distinct citations: 2.23
- Range citations: 2/30 (7%)
- Copied source headers: 2/30 (7%)
- Attack ID not in KB: 0/30 (0%)
- KB attack ID named: 9/30 (30%)
- SP not in KB: 0/30 (0%)
- Reports citing NIST source: 13/30 (43%)
- Reports citing MITRE source: 20/30 (67%)
- Class breakdowns:
  - Benign: n=10, words mean=320, with citation=6/10
  - DoS: n=10, words mean=393, with citation=10/10
  - DDoS: n=10, words mean=347, with citation=10/10

### Vanilla Mode (30 reports)
- Words min/mean/max: 294 / 384.3 / 488
- In 300–600 words: 29/30 (97%)
- All 5 numbered headings: 30/30 (100%)
- At least 1 inline citation: 0/30 (0%) (correct)
- Copied source headers: 0/30 (0%)
- Attack ID not in KB: 13/30 (43%) (hallucinations in vanilla)

## 2. Human Audit Results (36 judged citation rows across 10 flows)
- Full support ("yes"): 12/36 (33%)
- Partial support: 17/36 (47%)
- Unsupported ("no"): 7/36 (19%)
- Yes or partial: 29/36 (81%)

## 3. Configuration Parameters
- Model: `llama3:8b`
- Temperature: 0.2
- Seed: 42
- Context window: 8192
- Max tokens (`num_predict`): 1100
- Keep alive: 15m
- Min words: 300
- Max words: 600

