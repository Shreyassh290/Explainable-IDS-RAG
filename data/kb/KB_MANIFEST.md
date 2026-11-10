# Knowledge base manifest

Frozen on 2026-11-08 for the evaluation. Do not add, remove or edit documents.

- chunks.jsonl SHA-256: 876e30b4f9e3c6943ccc5c584b631f2316d20e10d27e012151ba8ed44a006d15
- total chunks: 742 | index chunk_count: 742 | index embedding dim: 768 | embed model: sentence-transformers/all-mpnet-base-v2
- index_meta chunks_sha256: 876e30b4f9e3c6943ccc5c584b631f2316d20e10d27e012151ba8ed44a006d15
- retrieval settings (config.yaml, rag section): bm25_b=0.75, bm25_k1=1.5, chunk_max_tokens=350, chunk_min_tokens=200, chunk_overlap=200, embed_batch_size=32, embed_device=auto, embed_model=sentence-transformers/all-mpnet-base-v2, fallback_min_chunks=3, fallback_score_threshold=0.5, fallback_sections=[{'source': 'NIST_SP_800-61_Rev2.pdf', 'section': 'Executive Summary'}, {'source': 'NIST_SP_800-61_Rev2.pdf', 'section': '3.6 Recommendations'}], final_k=5, fusion_top=30, fusion_weights={'semantic': 0.6, 'bm25': 0.4}, index_dir=data/kb/index, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, retrieve_k_each=30, rrf_k=60

## Documents

### MITRE_T1498.html
- SHA-256: 8c5c0ab7a556dcbf5b58605550c1b0717cc61e5fdaf359cf1c2c2de5e85ca2ac
- size: 50985 bytes | chunks: 5 | distinct sections: 4
- page title: Network Denial of Service, Technique T1498 - Enterprise | MITRE ATT&CK®
- ATT&CK version: 1.2 | created: 17 April 2019 | last modified: 24 October 2025

### MITRE_T1498_001.html
- SHA-256: 5a5ca77e45779381dd99938b8cc76d5304a414fbd26bfad24936e585904191e9
- size: 44011 bytes | chunks: 4 | distinct sections: 3
- page title: Network Denial of Service: Direct Network Flood, Sub-technique T1498.001 - Enterprise | MITRE ATT&CK®
- ATT&CK version: 1.4 | created: 02 March 2020 | last modified: 24 October 2025

### MITRE_T1499.html
- SHA-256: 95db680996a0b1d98a6d523adac35645d8ed1f35d091eb52e25aea5bafeccd5a
- size: 56885 bytes | chunks: 7 | distinct sections: 4
- page title: Endpoint Denial of Service, Technique T1499 - Enterprise | MITRE ATT&CK®
- ATT&CK version: 1.2 | created: 18 April 2019 | last modified: 24 October 2025

### NIST_SP_800-30_Rev1.pdf
- SHA-256: f214087f0bdb35932a28c16eb93932a33c67a6edfb5e6f1638866bd201c98e04
- size: 826897 bytes | chunks: 183 | distinct sections: 15
- pages: 95 | dates found on pages 1-3: September 2012
- first lines p1: I N F O R M A T I O N     S E C U R I T Y | Computer Security Division | Information Technology Laboratory | National Institute of Standards and Technology
- first lines p2: Special Publication 800-30                                                                               Guide for Conducting Risk Assessments | ___________________________________________________________________________
- first lines p3: Special Publication 800-30                                                                               Guide for Conducting Risk Assessments | ___________________________________________________________________________

### NIST_SP_800-61_Rev2.pdf
- SHA-256: b1c5ead0632dcfb979c59d8c462b094ad2465ddaf62d5c246289d6d9b4419df1
- size: 1744473 bytes | chunks: 201 | distinct sections: 55
- pages: 80 | dates found on pages 1-3: April 3, 2025, August 2012
- first lines p1: Date updated: April 3, 2025 | Withdrawn NIST Technical Series Publication | Warning Notice | The attached publication has been withdrawn (archived), and is provided solely for historical purposes.
- first lines p2: Computer Security | Incident Handling Guide | Recommendations of the National Institute | of Standards and T echnology
- first lines p3: NIST Special Publication 800-61 Revision 2 | Computer Security Incident Handling | Guide | Recommendations of the National

### NIST_SP_800-94.pdf
- SHA-256: 289c2ed27f7e3aac41624bf2e6fad268642bad2dabe352a5ab28fa169824a96e
- size: 1060429 bytes | chunks: 342 | distinct sections: 116
- pages: 127 | dates found on pages 1-3: February 2007
- first lines p1: Special Publication 800-94 | Guide to Intrusion Detection | and Prevention Systems | (IDPS)
- first lines p2: Guide to Intrusion Detection and | Prevention Systems (IDPS) | Recommendations of the National | Institute of Standards and Technology
- first lines p3: GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEMS (IDPS) | Reports on Computer Systems Technology | The Information Technology Laboratory (ITL) at the National Institute of Standards and Technology | (NIST) promotes th

## Content rules applied by chunking.py

- SP 800-61 Rev 2: table of contents, front matter, section 1.1 Authority, Appendix D and Appendix H dropped.
- SP 800-94: front matter, section 1.1 Authority, Appendix B, C and D dropped; Appendix A kept.
- SP 800-30 Rev 1: only chapters 1 to 3 kept; front matter and all appendices dropped.
- MITRE pages: navigation block and References section removed.
- Footnote numbers glued to words removed from the three PDFs (lists in artifacts/logs/footnote_subs_*.txt).
- Known limits: footnote text and figure text can remain inside chunks; PDF bullet glyphs are private-use characters.

## Differences from the paper's knowledge base

- Paper: 14 documents and 5,234 chunks. Here: the documents above and the chunk total above.
- NIST SP 800-30 is an addition (the paper cites its risk principles for the Threat Assessment section).
- SP 800-61 Rev 2 was withdrawn by NIST on 3 April 2025 (the PDF's own first page says so); it is used because the paper uses it.
- Embeddings are local (all-mpnet-base-v2), not an OpenAI API.
