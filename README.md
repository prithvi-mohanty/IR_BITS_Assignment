# IR Assignment 1 — End-to-End Information Retrieval System

**Course:** AIMLCZG537 / DSECLZG537 (Information Retrieval), BITS Pilani WILP  
**Semester:** 2, 2025-26  
**Deadline:** 15 June 2026, 23:59 PM  
**Framework:** Streamlit  

---

## Group 64

| Name | BITS ID | Contribution |
|------|---------|--------------|
| Prithvi Mohanty | 2025AA05707 | 34% |
| Monika Sharma | 2025AA05735 | 33% |
| Hanni Rajavikram | 2025AA05740 | 33% |

---

## Installation

```bash
pip install streamlit nltk pandas
```

NLTK data packages (punkt, stopwords, wordnet, averaged_perceptron_tagger) are downloaded
automatically on first run.

---

## Run

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## Dataset

The app includes a built-in corpus of 8 documents:
- 4 COVID-19 vaccine research paper excerpts (qualitative and survey studies from Ghana)
- 4 IR/NLP/ML concept documents covering inverted indexing, preprocessing, tolerant retrieval, and ranking

Custom datasets can be uploaded as `.txt` files via the **Section A** interface.

---

## Features

- [x] **A. Document upload** — built-in corpus or custom file upload with live viewer
- [x] **B. Text preprocessing** — tokenization, lowercasing, hyphen handling, stop word removal, Porter/Snowball stemming, WordNet lemmatization; inverted index construction; Jaccard similarity comparison (5 queries)
- [x] **C. Phrase query** — biword index and positional index with side-by-side results, false positive detection, batch comparison table
- [x] **D. BST vs B-Tree** — BST (random + sorted insertion) vs B-Tree benchmark; step counts + latency for 10–50 random queries; summary metrics and inference
- [x] **E. Tolerant retrieval** — wildcard k-gram search, Levenshtein spelling correction with DP matrix visualisation, Soundex phonetic matching with code table
- [x] **G. Discussion & Inference** — full written analysis for all components

---

## Submission Structure

```
IR_AIMLCZG537_Group64_Assignment1/
├── app.py
├── README.md
├── Report.pdf               ← export from Report_Outline.md + screenshots
└── screenshots/
    ├── 01_document_upload.png
    ├── 02_preprocessing_output.png
    ├── 03_inverted_index.png
    ├── 04_stemming_lemmatization_table.png
    ├── 05_jaccard_experiment.png
    ├── 06_phrase_query_results.png
    ├── 07_bst_btree_benchmark.png
    ├── 08_tolerant_wildcard.png
    ├── 09_tolerant_spelling.png
    ├── 10_tolerant_phonetic.png
    └── 11_virtual_lab.png    ← terminal + browser window with timestamp
```
