# ids-rag-mitigation

Implementation of "From Detection to Response: A Deep Learning and Retrieval-Augmented Generation Framework for Network Intrusion Mitigation" (title confirmed against the arXiv listing on 2026-11-10; authors Md Navid Bin Islam and Sajal Saha, University of Northern British Columbia). arXiv ID: 2605.17960 (v1, cs.CR, 18 May 2026).

The project reproduces the paper's pipeline locally with free tools (Phase 1) and reports honestly where the results do and do not match. Phase 2 (extension to more attack classes) is not done.

## What the pipeline does

1. CICIDS2018 flows are classified as Benign, DoS or DDoS by three binary DNNs (128-64-32, ReLU, dropout 0.3).
2. The top-5 features are chosen by the absolute gradient of the predicted-class logit and mapped to plain-language descriptions.
3. A query built from the class, features, protocol and destination port (never IP addresses) retrieves NIST and MITRE chunks: synonym expansion, 30 BM25 and 30 dense candidates, fusion (0.60 dense + 0.40 min-max-normalised BM25), cross-encoder rerank to the top 5, and a fallback that injects two SP 800-61 sections when fewer than 3 chunks score at least 0.5.
4. A five-section prompt (Rationale, Key Indicators, Confidence Assessment, Threat Assessment, Recommendations) goes to a local llama3:8b through Ollama (temperature 0.2, seed 42, num_ctx 8192).
5. RAG reports are compared with vanilla reports (same prompt, empty retrieval context).

All settings live in `config.yaml`. Seed is 42 everywhere.

## Setup (Windows PowerShell, uv only)

```powershell
cd C:\Users\shrey\Downloads\AIML_Project\ids-rag-mitigation
$env:PYTHONUTF8 = "1"
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"   # if uv is not found
uv sync
ollama pull llama3:8b
```

Embeddings (`sentence-transformers/all-mpnet-base-v2`) and the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) download from Hugging Face on first use. After they are cached, set `$env:HF_HUB_OFFLINE = "1"` for runs and unset it for downloads.

Dataset: the 10 processed CICIDS2018 CSVs from the public CSE-CIC-IDS2018 bucket on AWS (16,233,002 rows in total according to `artifacts\data_inspection.json`; the download size was not recorded) go in `data/raw/`. Only the Tuesday file contains IP addresses.

## Repository layout

- `config.yaml`, `src/ids_rag/config.py`
- `src/ids_rag/data/` inspect and preprocess
- `src/ids_rag/models/` DNN, training, ensemble and confidence tiers
- `src/ids_rag/explain/` feature dictionary, gradient and SHAP top-k
- `src/ids_rag/rag/` chunking, index, retrieve, manifest
- `src/ids_rag/llm/` prompt and generate
- `src/ids_rag/pipeline.py` end-to-end inference
- `src/ids_rag/eval/` classification, flow selection, retrieval, report quality, citation check
- `data/kb/` frozen knowledge base (documents, chunks.jsonl, KB_MANIFEST.md with hashes)
- `data/eval/` retrieval queries and the three reference reports
- `artifacts/eval/` evaluation outputs, audit sheet, key, rubric and results

## Running each stage

Each stage is a module under `src/ids_rag/`. Run one stage at a time and check its output before the next. Exact flags for the early stages: run `uv run python -m ids_rag.<module> --help`.

| Stage | Module | Command |
|---|---|---|
| 1 Data inspection and preprocessing | `ids_rag.data.inspect`, `ids_rag.data.preprocess` | `uv run python -m ids_rag.data.inspect`, then `uv run python -m ids_rag.data.preprocess` (no arguments; neither module has a command line). They need the raw CICIDS2018 CSV files, which are not included in this repository; the processed splits in `data\processed\` are. These two commands were not captured in shell history. |
| 2 Train three DNNs, ensemble | `ids_rag.models.train`, `ids_rag.models.ensemble` | `uv run python -m ids_rag.models.train --confirm-retrain`. Without `--confirm-retrain` the module prints a refusal and exits with code 2 without training (guard added on 10 Nov 2026 after an accidental retrain; `--help` is now safe for this module only). WARNING: with the flag it retrains the classifiers from scratch and overwrites `models\dnn_*.pt`, `models\feature_names.json` and the training histories and curves in `artifacts\`. `ids_rag.models.ensemble` is a library module with no command line. The temperatures in `models\temperatures.json` are fitted and written by `uv run python -m ids_rag.eval.classification`, which also rewrites the classification report, plots, predictions and metrics in `artifacts\`. The original training command was not captured in shell history. |
| 3 Explanations | `ids_rag.explain.shap_topk`, `ids_rag.explain.features` | `uv run python -m ids_rag.explain.shap_topk --examples` writes `artifacts\sample_explanations.json` and needs `artifacts\test_predictions.parquet` from the classification step. `--index N` explains one test row, `--k` sets the number of features, `--no-comparison` skips the SHAP comparison. `ids_rag.explain.features` is a library module. The original invocation was not captured in shell history. |
| 4 Knowledge base and index | `ids_rag.rag.chunking`, `ids_rag.rag.index`, `ids_rag.rag.manifest` | `uv run python -m ids_rag.rag.chunking`, then `uv run python -m ids_rag.rag.index --build`, then `uv run python -m ids_rag.rag.manifest` (writes `data\kb\KB_MANIFEST.md`). Optional checks: `uv run python -m ids_rag.rag.index --check` and `uv run python -m ids_rag.rag.retrieve --check`. |
| 5 Generation (one or more flows) | `ids_rag.pipeline` | below |

Pipeline options (from `--help`): `--index N` or `--indices 1,2,3` or `--indices-file FILE`; `--mode {rag,vanilla,both}`; `--no-generate` (stop after retrieval and write prompt files); `--seed`; `--out-dir`; `--skip-existing`; `--summary-csv`.

```powershell
uv run python -m ids_rag.pipeline --index 10045 --mode both
```

**Analyst alert (extension).** `ids_rag.alert` converts a stored `artifacts\reports\flows\flow_N.json` (written by the pipeline) into a short analyst alert, `artifacts\alerts\alert_flow_N.json`. The alert holds the predicted class and confidence, the top-5 features with exact values, the RAG report text, the numbered sources, a per-citation overlap flag (same 0.5 threshold as the citation check), validation flags, provenance and warnings. It leaves out the true label, the prompts and the config. It does not load models, does not call Ollama, and works only on flows the pipeline has already written for rows of the saved test set; it does not classify live traffic. It refuses to overwrite an existing alert unless `--force` is given.

```powershell
uv run python -m ids_rag.alert --index 0
```

Limits of the alert: the citation flag is a word-overlap heuristic. In the committed sample (flow 0) the sentence calling the attack "a sub-technique of T1498.001" is not flagged (overlap 0.778), although the cited chunk lists T1498.001 as a sub-technique of Network Denial of Service, and the report names only 2 of the 5 top features. The alert marks the second problem but cannot detect the first. It does not check whether the advice is correct.

## Frozen knowledge base

The knowledge base was frozen before evaluation: NIST SP 800-61 Rev 2 (withdrawn by NIST on 3 April 2025; used because the paper uses it), NIST SP 800-94, NIST SP 800-30 Rev 1 chapters 1 to 3 (an addition not in the paper's list), and MITRE ATT&CK T1498, T1498.001 and T1499. It contains 742 chunks of at most 350 tokens with 200-token overlap. Hashes are in `data/kb/KB_MANIFEST.md`, and a copy is in `backup/kb_frozen` (git-ignored).

## Evaluation

Select the 30 evaluation flows (10 per class, seed 42, seven tuning rows excluded; these settings come from `config.yaml`):

```powershell
uv run python -m ids_rag.eval.select_flows            # refuses to overwrite without --force
```

Options: `--test-path`, `--label-map-path`, `--out-path`, `--seed`, `--per-class`, `--exclude-rows` (default: the `eval.exclude_rows` list in `config.yaml`, which is 0, 3, 4, 637, 100, 2500, 60000), `--force`. The saved selection `artifacts\eval\eval_flows.json` is identical to the copy in `backup\eval_snapshot.zip`, including its `created_at` timestamp (2026-11-09T10:09:23 UTC).

Generate RAG and vanilla reports for the 30 flows: Recorded run: `uv run python -m ids_rag.eval.select_flows` (seed 42, 10 flows per class, exclusions taken from `config.yaml`), then `uv run python -m ids_rag.pipeline --indices-file artifacts\eval\eval_flows.json --mode both --out-dir artifacts\eval\flows --summary-csv artifacts\eval\flow_summary.csv` (17.86 minutes wall time; 30 flows run, 0 skipped, 0 failed). Config used for that run: `artifacts/eval/config_used_for_eval_run.yaml` (identical to the `config.yaml` committed in `66b502c`; the working copy of `config.yaml` now adds one line, `prompt_version: tuned`, so reproducing these frozen results needs `prompt_version: baseline`, which is also the code default when the key is absent).

Aggregates and the blind citation audit:

```powershell
uv run python artifacts\eval\aggregate.py
uv run python artifacts\eval\audit_sheet_v2.py        # writes sheet, key and rubric; refuses to overwrite
uv run python artifacts\eval\audit_score_v2.py --out artifacts\eval\audit_results_v2.txt
uv run pytest -q                                      # 128 tests
```

Retrieval and report-quality evaluation live in `ids_rag.eval.retrieval` and `ids_rag.eval.report_quality` Recorded runs: `uv run python -m ids_rag.eval.retrieval` and `uv run python -m ids_rag.eval.report_quality` (no flags; `--no-bert` skips BERTScore).

## Results

All numbers below come from recorded runs and terminal output, not from the paper.

**Classification (CICIDS2018 test split, 119,454 flows).** Accuracy 0.9984 (paper: 0.9984). Per-class F1: Benign 0.9979, DoS 0.9992, DDoS 0.9985. Expected calibration error 0.0011 raw.

**Retrieval (33 queries written by the project author).**

| Metric | This project | Paper |
|---|---|---|
| Precision@5 | 0.539 (ceiling 0.83) | 0.91 |
| Recall@5 (share of relevant chunks found in the top 5) | 0.499 (ceiling 0.708) | 0.84 |
| Section recall (share of relevant sections hit; not the metric the paper reports) | 0.818 | not reported |
| MRR | 0.892 | 0.87 |
| Success rate | 0.970 | 0.97 |

Query 11 (slow-rate attacks that hold web server connections open) is the one failed query: none of its top 5 chunks is labelled relevant. A wording search of all 742 knowledge-base chunks for slowloris-style terms (for example slowloris, slow HTTP, low-and-slow, half-open, keep-alive) found no matches, so the knowledge base appears to hold no text on connection-holding attacks. The relevance label for this query is the four T1499 Description chunks, the nearest general section; none of the four mentions connections, sessions or slowness, so this query measures a gap in the knowledge base and not a retrieval error. SP 800-94 does discuss slow, low-volume activity that evades anomaly detection (sections 2.3.2 and 6.2.3), which is related but not the same thing. The search is by wording, so a paraphrase could have been missed. The paper names slow-rate attacks as future work.

**Generated reports (30 flows, all classified correctly).**

- Words: RAG mean 353.5 (min 214, max 415); 29 of 30 inside 300-600; 4 of 30 inside 400-500.
- Reports with at least one inline citation: RAG 26 of 30 (Benign 6, DoS 10, DDoS 10); vanilla 0 of 30.
- ATT&CK ids: 9 of 30 RAG reports name an id from the knowledge base and none names an id outside it. 13 of 30 vanilla reports name an id that is not in the knowledge base (not individually verified as wrong).

**Citation support audit (10 RAG reports, 36 citation rows, single scorer).** Each citation was judged against the full text of the chunk it cites, using `artifacts/eval/audit_rubric_v2.txt`.

| Class | Rows (flows) | yes | partial | no |
|---|---|---|---|---|
| Benign | 5 (3) | 0 | 4 | 1 |
| DoS | 18 (4) | 9 | 6 | 3 |
| DDoS | 13 (3) | 3 | 7 | 3 |
| All | 36 (10) | 12 | 17 | 7 |

Overall 12 of 36 citations (33%) fully supported, 29 of 36 (81%) at least partly. By source: NIST 3 of 9 yes, MITRE 9 of 27 yes. ATT&CK ids in the audited sample: 3 of 3 correct (all from two DoS reports; the sample has no DDoS id rows). The paper reports citation presence rates (NIST 93.3%, MITRE 86.7%), not whether citations support their sentences, so the two are not directly comparable.

**Citation check (extension, annotation only).** The Phase 1 reports are not regenerated. A post-processing check flags each inline citation as "unverified" when the word overlap between its sentence and the cited chunk is below 0.5 (`citation_check.overlap_threshold` in `config.yaml`). Copied source-header lines are skipped, as in report generation. The checker reproduces all 36 manually audited rows (36 of 36 matched, maximum overlap difference 0.0). The checker is `src/ids_rag/eval/citation_check.py` (tests in `tests/test_citation_check.py`). Run it with `uv run python -m ids_rag.eval.citation_check --self-check`; it writes `citation_check_rows.csv`, `citation_check_per_report.csv` and `citation_check_summary.txt` to `artifacts/eval/citation_check/` and refuses to overwrite existing files. It was ported from an earlier scratch script, and its two CSV outputs are byte-identical to that script run.

Descriptive result over all 30 RAG reports: 68 of 106 unique citations (64.2%) are flagged, and none cites a chunk outside the retrieved list. By true class: Benign 9 of 9, DoS 35 of 57, DDoS 24 of 40. By source: NIST 12 of 18, MITRE 56 of 88. In the 10 audited flows 20 of 36 are flagged; in the other 20 flows 48 of 70 (68.6%).

Against the 36 audit judgments, kept (not flagged) at 0.5 were yes 11 of 12, partial 5 of 17, no 0 of 7. AUC of the overlap score is about 0.94 for yes versus partial+no and about 0.71 for yes+partial versus no. The threshold was chosen with these rows in view, so they cannot validate it.

Blind check on 16 further rows from 8 unaudited flows (B01 to B08: 8 DoS, 4 DDoS, 4 Benign rows). The sheet hid overlap, flags and class.

| Judgment | Rows | Kept | Flagged |
|---|---|---|---|
| yes | 7 | 6 | 1 |
| partial | 5 | 1 | 4 |
| no | 4 | 0 | 4 |

8 of 9 flagged rows were partial or no, and 6 of 7 kept rows were yes. AUC was 0.905 for yes versus partial+no and 0.979 for yes+partial versus no (the latter rests on only 4 "no" rows). A flag means "not clearly supported by the chunk", not "wrong". The score separates fully supported citations from the rest but does not reliably separate partial from no. This is a first check on a small sample, not a validation.

**RAG versus vanilla quality (BERTScore, ROUGE, BLEU, paired Wilcoxon with Holm correction).** No reliable difference. BERTScore F1 0.8610 (RAG) against 0.8617 (vanilla), a difference of -0.0006 (RAG higher on 12 of 30 flows, lower on 18); every Holm-adjusted p is 0.526 or higher. ROUGE-1 and ROUGE-2 are slightly higher for RAG (+2.7% and +7.1% relative) and ROUGE-L and BLEU slightly lower, none significant. The three class references are shared by 10 flows each, so the 30 pairs are not independent. The paper reports large gains (for example BERTScore F1 +4.4%, ROUGE-2 +126.4%). This result is not reproduced here.

Overlap diagnostic: 6.7% of RAG report trigrams appear in the retrieved chunks, against 1.6% for vanilla, and 15 of 30 RAG reports copy an 8-word span. The three reference reports share 21%, 29% and 30% of their trigrams with the knowledge base (Benign, DoS, DDoS) and contain verbatim 8-word spans from it, so they are not independent of the retrieved text (see Limitations).

## Limitations

- Only correctly classified flows were evaluated, on one dataset, with 30 flows. Retrieval depends on the predicted class: over 10 flows per class there were 5, 2 and 2 distinct top-5 chunk sets (Benign, DoS, DDoS), so reports within a class are not independent and the repeated sentences are expected.
- The 33 retrieval queries, the audit judgments and the three reference reports were written or made by the project author. The references overlap the knowledge base: they share 21%, 29% and 30% of their word trigrams with it (Benign, DoS, DDoS) and contain verbatim 8-word spans from it (10, 2 and 17 respectively), so they are not independent of the retrieved text. The paper says its references were merged from its sources, which could explain its larger gains; this is an untested inference.
- The audit has one scorer who also built the system, and blinding is partial (sentences reveal the class). No inter-scorer agreement was computed. Rows cluster within flows and the sample is small, so class differences (DoS 9 of 18 yes against DDoS 3 of 13) are suggestive only. The "partial" label includes citations that support only a general principle while the flow-specific content comes from the prompt; a different rubric would change the yes rate.
- Instruction following: the prompt forbids ranges and reference lists, but 5 of 30 RAG reports use a list or range form such as [2, 3] (author's regex count; the stored range metric in `aggregate.txt` reports 2 of 30), and 2 of 30 copy source headers. One report (flow 22251) classified DoS also calls the attack likely DDoS.
- The three Benign reports without any citation are 10947, 24001 and 92199. The only bracket in flow 52197 is a reference-list line, not an inline citation.
- Differences from the paper: 41 features instead of 66; 796,358 rows instead of about 1.15M; 6 knowledge-base documents instead of 14 (742 chunks instead of 5,234); local embeddings instead of OpenAI (the paper is inconsistent on this); SP 800-30 added; Benign synonym expansion added; near-average feature implications replaced by a neutral line (`suppress_near_average_implications: true`); a citation rule added to the prompt. The paper also gives two report structures (five sections in the template, four blocks in Fig. 7) and two lengths (300 to 600 and 400 to 500 words); this project follows the template and 300 to 600.
- SHAP GradientExplainer was compared with plain gradient attribution (mean top-5 overlap 1.6 of 5); the gradient method is used because the paper's Equation 9 specifies it.
- Repeat test (9 Nov 2026, files in backup\reports_run_a): 8 report files (Benign, DoS, DDoS and one misclassified case, each RAG and vanilla) were generated three times; 6 of 8 were byte-identical across all three runs, and runs 2 and 3 matched each other in 8 of 8. The options recorded for that test were llama3:8b, temperature 0.2, seed 42, num_ctx 8192 and num_predict 1100. The two differing sets are the misclassified case, where run 1 differs from runs 2 and 3; the cause was not investigated. Reproducibility was not re-measured on the 30-flow evaluation set, so identical output on a rerun is not guaranteed.
- Model files: the Benign classifier was accidentally retrained on 10 Nov 2026 (a looped `--help` call started a real training run). The retrained `models\dnn_Benign.pt` is byte-identical to the file evaluated earlier (hash recorded before the evaluation run in `artifacts\eval\hashes_at_eval.txt`), and rerunning the classification step gave 0 of 119,454 predictions different. The original Benign training history and curves were overwritten, so the files in `artifacts\` are the regenerated ones. SHA-256 hashes of the model and key data files are in `MODEL_MANIFEST.md`.
- ATT&CK technique names were checked against the knowledge base only for T1498, T1498.001, T1499 and T1499.001. Seven id-naming reports outside the audit sample were read but not formally scored. The BERTScore layer choice (distilroberta-base, layer 5) is the author's.
- The citation flag is a word-overlap heuristic, not a verdict. Its threshold was chosen with the 36 audited rows in view. The blind check covers 16 rows from 8 flows, judged by one scorer who built the system, with partial blinding and clustered rows. The other 20 blind rows (B09 to B16) are unscored.

## Future work

- A second scorer on at least 15 audit rows to compute Cohen's kappa (script already supports it: `--second`).
- A scored id table for the seven id-naming reports outside the audit sample.
- Rescore against references merged from knowledge-base sections chosen in advance, as a test of the paper's reference construction.
- Rerun the 30 flows with `suppress_near_average_implications: false` and with grad-times-input attribution.
- Phase 2: add Bot and BruteForce classes (needs preprocessing, more DNNs, MITRE T1110 and related documents, new queries and a fresh evaluation).
- UNSW-NB15 generalization (the paper reports 95.3%); requires a large download.