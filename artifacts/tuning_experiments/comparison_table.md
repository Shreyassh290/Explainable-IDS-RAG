# Pilot Tuning Experiments — Comparison Report

This report compares the 7 representative pilot flows across Baseline, Experiment B (Initial Prompt Tuning), and Experiment B v2 (Revised Grounding & Anti-Hallucination).

## Pilot Flow Set Overview

| Flow ID | Class | Selection Reason |
|---|---|---|
| 10045 | Benign | Clean baseline benign flow; tests regression prevention |
| 52197 | Benign | Hardest benign case (214 words, copied source header in baseline) |
| 44644 | DoS | Standard DoS flow with 4 citations; tests citation grounding |
| 22251 | DoS | Baseline added invalid References block at end |
| 106513 | DDoS | Baseline hallucinated botnet claim |
| 8136 | DDoS | Baseline omitted numbered headings |
| 115902 | DDoS | Metadata vs attack evidence grounding check |

## Automated Metrics Comparison (RAG Mode)

| Flow ID | Class | Metric | Baseline | Exp B (Initial) | Exp B v2 (Revised) | Target / Ideal |
|---|---|---|---|---|---|---|
| 10045 | Benign | Words (300-600) | 324 | 401 | 391 | 300–600 |
| 10045 | Benign | Headings (5/5) | 5/5 | 5/5 | 5/5 | 5/5 |
| 10045 | Benign | In-text citations | [1] | [1] | [none] | Grounded |
| 10045 | Benign | Copied/Ref list | No | No | No | No |
|---|---|---|---|---|---|---|
| 52197 | Benign | Words (300-600) | 214 | 408 | 391 | 300–600 |
| 52197 | Benign | Headings (5/5) | 5/5 | 5/5 | 5/5 | 5/5 |
| 52197 | Benign | In-text citations | [none] | [1] | [none] | Grounded |
| 52197 | Benign | Copied/Ref list | Yes | No | No | No |
|---|---|---|---|---|---|---|
| 44644 | DoS | Words (300-600) | 388 | 407 | 414 | 300–600 |
| 44644 | DoS | Headings (5/5) | 5/5 | 5/5 | 5/5 | 5/5 |
| 44644 | DoS | In-text citations | [1;2;3;4] | [1;2;3] | [1;2] | Grounded |
| 44644 | DoS | Copied/Ref list | No | No | No | No |
|---|---|---|---|---|---|---|
| 22251 | DoS | Words (300-600) | 413 | 479 | 351 | 300–600 |
| 22251 | DoS | Headings (5/5) | 5/5 | 5/5 | 5/5 | 5/5 |
| 22251 | DoS | In-text citations | [1;2] | [1] | [1;2] | Grounded |
| 22251 | DoS | Copied/Ref list | Yes | No | No | No |
|---|---|---|---|---|---|---|
| 106513 | DDoS | Words (300-600) | 335 | 475 | 416 | 300–600 |
| 106513 | DDoS | Headings (5/5) | 5/5 | 5/5 | 5/5 | 5/5 |
| 106513 | DDoS | In-text citations | [1;2;3;5] | [none] | [1;2] | Grounded |
| 106513 | DDoS | Copied/Ref list | No | Yes | No | No |
|---|---|---|---|---|---|---|
| 8136 | DDoS | Words (300-600) | 335 | 391 | 440 | 300–600 |
| 8136 | DDoS | Headings (5/5) | 0/5 | 5/5 | 5/5 | 5/5 |
| 8136 | DDoS | In-text citations | [1;5] | [1] | [1;2] | Grounded |
| 8136 | DDoS | Copied/Ref list | No | No | No | No |
|---|---|---|---|---|---|---|
| 115902 | DDoS | Words (300-600) | 336 | n/a | 349 | 300–600 |
| 115902 | DDoS | Headings (5/5) | 5/5 | n/a/5 | 5/5 | 5/5 |
| 115902 | DDoS | In-text citations | [1;2;3;4;5] | [none] | [2] | Grounded |
| 115902 | DDoS | Copied/Ref list | No | n/a | No | No |
|---|---|---|---|---|---|---|