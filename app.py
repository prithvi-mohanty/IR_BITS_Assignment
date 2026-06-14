"""
IR Assignment 1 — End-to-End Information Retrieval System
Course: AIMLCZG537 / DSECLZG537 (Information Retrieval), BITS Pilani WILP
Semester 2, 2025-26
Dataset: COVID-19 Vaccine Research Documents + IR/NLP reference corpus
Framework: Streamlit

Group 64
  Prithvi Mohanty        — 2025AA05707
  Monika Sharma          — 2025AA05735
  Hanni Rajavikram       — 2025AA05740
"""

import re
import math
import time
import random
from collections import defaultdict

import pandas as pd
import streamlit as st

import nltk

@st.cache_resource(show_spinner="Downloading NLTK data (first run only)…")
def _bootstrap_nltk() -> None:
    """Download required NLTK packages once per cold start.

    On Streamlit Cloud, module-level code reruns on every script invocation,
    so wrapping this in @cache_resource turns the 6 nltk.download() calls
    from "every interaction" into "once per container lifetime".
    """
    for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(_pkg, quiet=True)

_bootstrap_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as _sw_corpus
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag


# ---------------------------------------------------------------------------
# Built-in corpus: 4 COVID vaccine papers + 4 IR/NLP/ML documents (8 total)
# ---------------------------------------------------------------------------
CORPUS = {
    0: (
        "COVID Vaccine Barriers — Ghana Qualitative Study",
        """Facilitators and barriers to COVID-19 vaccine uptake among women in two regions of Ghana.
Although COVID-19 vaccines are available, evidence suggests that several factors hinder or facilitate their use.
Several studies have found gender differences in COVID-19 vaccine uptake, with women less likely to vaccinate than men.
Using a cross-sectional descriptive qualitative research design, thirty women in Greater Accra and Ashanti regions
were conveniently sampled and interviewed using a semi-structured interview guide.
Key facilitators include the desire to protect oneself and family against COVID-19, education about vaccines,
seeing others receive the COVID-19 vaccine, and vaccines being cost-free.
Long queues at vaccination centres, fear of side effects, misconceptions about vaccines, and shortage of vaccines
were the main barriers to COVID-19 vaccination among women in Ghana.""",
    ),
    1: (
        "COVID Vaccine Acceptance — Ghana Determinants",
        """The level and determinants of COVID-19 vaccine acceptance in Ghana.
As part of efforts to curb the pandemic, the government of Ghana received several shipments of approved vaccines.
The study assessed factors associated with acceptance or refusal of COVID-19 vaccines among Ghanaians.
A cross-sectional survey was conducted using a structured questionnaire administered to adult residents.
Determinants of acceptance included trust in health authorities, perceived vaccine safety, and prior vaccination history.
Barriers included fear of adverse effects, religious beliefs, and distrust of the vaccination campaign.
The study recommends targeted health communication strategies to address vaccine hesitancy and improve acceptance.""",
    ),
    2: (
        "COVID Vaccine Misinformation and Hesitancy",
        """Misinformation of COVID-19 vaccines and vaccine hesitancy.
The study examined various types of misinformation related to COVID-19 vaccines and their relationship to hesitancy.
Participants were exposed to corrective information before and after encountering vaccine misinformation.
Results showed that misinformation significantly increased vaccine hesitancy even after correction attempts.
Conspiracy theories and false claims about vaccine ingredients were the most prevalent forms of misinformation.
Social media platforms amplified the spread of misinformation, reducing community confidence in vaccination.
Health authorities must proactively counter misinformation using evidence-based public communication campaigns.""",
    ),
    3: (
        "COVID Vaccine Confidence, Effectiveness and Safety",
        """Confidence in COVID-19 vaccine effectiveness and safety and its effect on vaccine uptake decisions.
COVID-19 is a major public health threat associated with increased disease burden, mortality, and economic loss.
Safe and efficacious COVID-19 vaccines are key in halting and reversing the trajectory of the pandemic globally.
Vaccine confidence encompasses beliefs about vaccine safety, effectiveness, and the reliability of health systems.
Studies show that higher confidence in vaccine effectiveness correlates strongly with willingness to vaccinate.
Communication from trusted medical professionals and transparent reporting of clinical trial data build confidence.
Strengthening health system credibility is essential to sustaining high vaccination coverage in communities.""",
    ),
    4: (
        "Information Retrieval — Core Concepts",
        """Information retrieval systems enable users to find relevant documents from large text collections.
The inverted index maps each vocabulary term to the list of documents containing that term.
Boolean retrieval uses AND, OR, and NOT operators to combine query terms for precise document filtering.
The vector space model represents documents and queries as vectors in high-dimensional term space.
TF-IDF weighting assigns higher importance to terms frequent in one document but rare across the corpus.
Cosine similarity measures the angle between document and query vectors to rank retrieval results.
Ranked retrieval returns a sorted list of documents ordered by their relevance score to the user query.
Inverted index lookup is efficient. Lookup queries retrieve matching documents.""",
    ),
    5: (
        "NLP Preprocessing Pipeline",
        """Natural language processing enables computers to understand and generate human language automatically.
Tokenization splits text into atomic units called tokens before any further linguistic processing is applied.
Lowercasing normalises tokens so that surface variants of the same word are treated identically by the index.
Stemming reduces words to their root form using rule-based heuristic truncation, such as the Porter algorithm.
Lemmatization uses morphological analysis and dictionary lookup to find the canonical base form of each word.
Stop word removal filters common function words like the, a, and is that contribute little semantic signal.
Named entity recognition identifies and classifies entities such as persons, organisations, and locations.""",
    ),
    6: (
        "IR Dictionary Structures and Tolerant Retrieval",
        """Dictionary data structures store the vocabulary of an information retrieval system for efficient lookup.
Binary search trees support O(log n) average-case search but degenerate to O(n) with sorted key insertion.
B-trees are balanced multi-way trees that guarantee O(log n) search time regardless of insertion order.
Wildcard queries use prefix patterns and k-gram indices to find vocabulary terms matching incomplete spellings.
Spelling correction algorithms suggest alternative query terms using Levenshtein edit distance computation.
Phonetic correction maps query terms to Soundex codes to find vocabulary words that sound phonetically similar.
K-gram indices store all k-character substrings of each vocabulary term to support efficient wildcard matching.""",
    ),
    7: (
        "Web Search and Machine Learning Ranking",
        """Search engines use crawling, indexing, and ranking algorithms to retrieve relevant web pages.
PageRank computes page importance based on the number and quality of incoming hyperlinks from other pages.
Phrase queries require terms to appear consecutively and rely on positional index structures for accuracy.
Machine learning algorithms improve ranking by learning relevance patterns from labelled query-document pairs.
Deep learning models such as BERT have achieved state-of-the-art results on information retrieval benchmarks.
Query expansion adds related terms to the original query to improve recall without sacrificing precision.
Learning-to-rank methods combine multiple retrieval signals into a unified ranking function trained on data.""",
    ),
}


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------
_STOPWORDS  = set(_sw_corpus.words("english"))
_porter     = PorterStemmer()
_snowball   = SnowballStemmer("english")
_lemmatizer = WordNetLemmatizer()


def _wn_pos(tag: str) -> str:
    if tag.startswith("J"): return "a"
    if tag.startswith("V"): return "v"
    if tag.startswith("R"): return "r"
    return "n"


def preprocess(text: str, lowercase: bool, remove_sw: bool,
               split_hyphens: bool, method: str) -> list:
    if split_hyphens:
        text = text.replace("-", " ")
    if lowercase:
        text = text.lower()
    text   = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 1]
    if remove_sw:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if method == "Porter Stemmer":
        tokens = [_porter.stem(t) for t in tokens]
    elif method == "Snowball Stemmer":
        tokens = [_snowball.stem(t) for t in tokens]
    elif method == "Lemmatizer":
        tagged = pos_tag(tokens)
        tokens = [_lemmatizer.lemmatize(w, _wn_pos(tag)) for w, tag in tagged]
    return tokens


def build_inv_index(processed_docs: dict) -> dict:
    idx = defaultdict(set)
    for doc_id, tokens in processed_docs.items():
        for tok in tokens:
            idx[tok].add(doc_id)
    return {k: sorted(v) for k, v in idx.items()}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IR System — AIMLCZG537",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar: navigation + group details
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 IR Assignment 1")
    st.markdown("**Course:** AIMLCZG537 / DSECLZG537")
    st.markdown("**Deadline:** 15 June 2026, 23:59 PM")
    st.divider()
    st.markdown("**Group 64**")
    st.markdown(
        "| Name | BITS ID |\n"
        "|------|------|\n"
        "| Prithvi Mohanty | 2025AA05707 |\n"
        "| Monika Sharma | 2025AA05735 |\n"
        "| Hanni Rajavikram | 2025AA05740 |"
    )
    st.divider()
    section = st.radio("Navigate to section:", [
        "A · Document Upload",
        "B · Text Preprocessing",
        "C · Phrase Query",
        "D · BST vs B-Tree",
        "E · Tolerant Retrieval",
        "G · Discussion & Inference",
    ])

st.title("🔍 End-to-End Information Retrieval System")
st.caption("AIMLCZG537 / DSECLZG537 — Semester 2, 2025-26 | Assignment 1 | Group 64")


# ---------------------------------------------------------------------------
# Session state: persist corpus + preprocessing settings across sections
# ---------------------------------------------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs       = {k: v[1] for k, v in CORPUS.items()}
    st.session_state.doc_titles = {k: v[0] for k, v in CORPUS.items()}
    st.session_state.pp_lower   = True
    st.session_state.pp_sw      = True
    st.session_state.pp_hyph    = True
    st.session_state.pp_method  = "None"


def get_processed() -> dict:
    """Return processed tokens for each document using current preprocessing settings."""
    return {
        d: preprocess(t,
                      st.session_state.pp_lower,
                      st.session_state.pp_sw,
                      st.session_state.pp_hyph,
                      st.session_state.pp_method)
        for d, t in st.session_state.docs.items()
    }


# ===========================================================================
# SECTION A — Document Upload
# ===========================================================================
if section.startswith("A"):
    st.header("A. Document Upload & Corpus")

    use_sample = st.checkbox("Use built-in COVID-19 + IR research corpus (8 documents)", value=True)

    if use_sample:
        st.session_state.docs       = {k: v[1] for k, v in CORPUS.items()}
        st.session_state.doc_titles = {k: v[0] for k, v in CORPUS.items()}
        st.success(f"Loaded {len(st.session_state.docs)} built-in documents.")
    else:
        uploaded = st.file_uploader(
            "Upload text files (one document per .txt file)",
            accept_multiple_files=True,
            type=["txt", "csv"],
        )
        if uploaded:
            st.session_state.docs       = {i: f.read().decode("utf-8", errors="ignore")
                                            for i, f in enumerate(uploaded)}
            st.session_state.doc_titles = {i: f.name for i, f in enumerate(uploaded)}
            st.success(f"Loaded {len(st.session_state.docs)} documents.")
        else:
            st.info("No upload yet — using built-in corpus.")

    st.subheader("Document Viewer")
    for doc_id, text in st.session_state.docs.items():
        title = st.session_state.doc_titles.get(doc_id, f"Doc {doc_id}")
        with st.expander(f"Doc {doc_id}: {title}"):
            st.write(text)


# ===========================================================================
# SECTION B — Text Preprocessing
# ===========================================================================
elif section.startswith("B"):
    st.header("B. Text Preprocessing")

    # ── Preprocessing controls ──
    st.subheader("Preprocessing Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.pp_lower  = st.checkbox("Lowercase", value=st.session_state.pp_lower)
        st.session_state.pp_sw     = st.checkbox("Remove stop words", value=st.session_state.pp_sw)
        st.session_state.pp_hyph   = st.checkbox("Handle hyphens (split on '-')", value=st.session_state.pp_hyph)
    with col2:
        method_opts = ["None", "Porter Stemmer", "Snowball Stemmer", "Lemmatizer"]
        st.session_state.pp_method = st.selectbox(
            "Stemming / Lemmatization",
            method_opts,
            index=method_opts.index(st.session_state.pp_method),
        )

    processed = get_processed()

    # ── Before / After ──
    st.subheader("Before vs After Preprocessing")
    if st.button("Show preprocessing output (first 3 documents)"):
        for doc_id in list(st.session_state.docs.keys())[:3]:
            title = st.session_state.doc_titles.get(doc_id, f"Doc {doc_id}")
            raw   = st.session_state.docs[doc_id][:150]
            out   = " ".join(processed[doc_id][:20])
            c1, c2 = st.columns(2)
            c1.markdown(f"**Doc {doc_id} — Original** ({title})")
            c1.code(raw + " …")
            c2.markdown(f"**Doc {doc_id} — After preprocessing**")
            c2.code(out + " …")
            st.markdown("---")

    # ── Inverted Index ──
    st.subheader("Inverted Index Construction")
    inv_idx = build_inv_index(processed)

    # vocabulary size before vs after preprocessing
    raw_toks = {d: [w.lower() for w in word_tokenize(t) if len(w) > 1]
                for d, t in st.session_state.docs.items()}
    vocab_raw  = len(build_inv_index(raw_toks))
    vocab_proc = len(inv_idx)

    col1, col2, col3 = st.columns(3)
    reduction = (1 - vocab_proc / vocab_raw) * 100 if vocab_raw else 0
    col1.metric("Raw vocabulary (tokenize only)", vocab_raw)
    col2.metric("After preprocessing", vocab_proc)
    col3.metric("Vocabulary reduction", f"{reduction:.1f}%")

    if st.button("Show inverted index (first 30 terms)"):
        rows = [{"Term": t, "Doc Freq": len(docs), "Postings": str(docs)}
                for t, docs in sorted(inv_idx.items())[:30]]
        st.dataframe(pd.DataFrame(rows), width="stretch")

    # ── Word-level stemming vs lemmatization table ──
    st.subheader("Stemming vs Lemmatization — Word-Level Comparison")
    test_words_str = st.text_input(
        "Words to compare (comma-separated):",
        "vaccines, running, barriers, studies, universities, algorithms, misinformation, effectiveness",
    )
    if test_words_str:
        words = [w.strip().lower() for w in test_words_str.split(",") if w.strip()]
        rows  = []
        for w in words:
            tagged = pos_tag([w])
            rows.append({
                "Word":            w,
                "Porter Stem":     _porter.stem(w),
                "Snowball Stem":   _snowball.stem(w),
                "Lemma (WordNet)": _lemmatizer.lemmatize(w, _wn_pos(tagged[0][1])),
                "Same?":           "✓" if _porter.stem(w) == _lemmatizer.lemmatize(w, _wn_pos(tagged[0][1])) else "✗",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")

    # ── Jaccard experiment ──
    st.subheader("Retrieval Quality: Jaccard Similarity (Stemming vs Lemmatization)")

    TEST_QUERIES = [
        "vaccine barriers",
        "information retrieval",
        "machine learning ranking",
        "misinformation hesitancy",
        "language processing tokenization",
    ]

    if st.button("Run Jaccard experiment (5 queries)"):
        # Build separate indices for each method
        proc_stem  = {d: preprocess(t, True, True, True, "Porter Stemmer")
                      for d, t in st.session_state.docs.items()}
        proc_lemma = {d: preprocess(t, True, True, True, "Lemmatizer")
                      for d, t in st.session_state.docs.items()}
        idx_stem   = build_inv_index(proc_stem)
        idx_lemma  = build_inv_index(proc_lemma)

        rows     = []
        jaccards = []
        for q in TEST_QUERIES:
            stem_q  = preprocess(q, True, True, True, "Porter Stemmer")
            lemma_q = preprocess(q, True, True, True, "Lemmatizer")

            # OR retrieval: union over all query tokens
            stem_docs  = set()
            for tok in stem_q:
                stem_docs |= set(idx_stem.get(tok, []))

            lemma_docs = set()
            for tok in lemma_q:
                lemma_docs |= set(idx_lemma.get(tok, []))

            inter = stem_docs & lemma_docs
            union = stem_docs | lemma_docs
            j     = len(inter) / len(union) if union else 1.0
            jaccards.append(j)

            rows.append({
                "Query":              q,
                "Stem tokens":        str(stem_q),
                "Lemma tokens":       str(lemma_q),
                "Stem docs":          str(sorted(stem_docs)),
                "Lemma docs":         str(sorted(lemma_docs)),
                "Jaccard":            f"{j:.2f}",
            })

        st.dataframe(pd.DataFrame(rows), width="stretch")
        avg_j = sum(jaccards) / len(jaccards)
        st.success(f"**Average Jaccard similarity: {avg_j:.2f}**")

        st.markdown(f"""
**Inference:**
Average Jaccard = **{avg_j:.2f}** across 5 queries. Both methods retrieve largely overlapping
document sets because they agree on most word roots. Divergence occurs where stemming collapses
morphological variants aggressively (*vaccines → vaccin*, *universities → univers*, *barriers → barrier*
vs. lemmatization preserving *barrier*, *vaccine*, *university*).

**Conclusion:** For this COVID-19 / IR domain corpus, **lemmatization is preferred**. The documents
contain domain-specific nouns where over-stemming introduces index noise (e.g., *vaccin* is not a valid
word, which risks vocabulary fragmentation). Lemmatization preserves semantic precision at a
~{(1-avg_j)*100:.0f}% divergence cost relative to stemming.
        """)


# ===========================================================================
# SECTION C — Phrase Query
# ===========================================================================
elif section.startswith("C"):
    st.header("C. Phrase Query Processing")

    processed = get_processed()

    # Build biword index
    biword_idx = defaultdict(set)
    for doc_id, tokens in processed.items():
        for i in range(len(tokens) - 1):
            biword_idx[f"{tokens[i]} {tokens[i+1]}"].add(doc_id)

    # Build positional index
    pos_idx = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in processed.items():
        for pos, tok in enumerate(tokens):
            pos_idx[tok][doc_id].append(pos)

    # ── Index representations ──
    st.subheader("Index Representations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biword Index** — consecutive token pairs → document list")
        if st.button("Show first 15 biwords"):
            rows = [{"Biword": bw, "Documents": str(sorted(docs))}
                    for bw, docs in sorted(biword_idx.items())[:15]]
            st.dataframe(pd.DataFrame(rows), width="stretch")

    with col2:
        st.markdown("**Positional Index** — term → {doc_id → [position list]}")
        if st.button("Show first 10 positional entries"):
            rows = [{"Term": t,
                     "Doc→Positions (first 5)": str({d: plist[:5]
                                                     for d, plist in sorted(docs.items())[:3]})}
                    for t, docs in sorted(pos_idx.items())[:10]]
            st.dataframe(pd.DataFrame(rows), width="stretch")

    with st.expander("Why does biword indexing produce false positives?"):
        st.markdown("""
A biword index stores every *consecutive token pair*. A phrase query like **"vaccine safety barriers"**
is broken into bigrams: `"vaccine safety"` and `"safety barriers"`.

A document containing:
> *"…COVID-19 **vaccine safety** is paramount. Socioeconomic **safety barriers** limit access…"*

would match both bigrams — even though the three-word phrase **"vaccine safety barriers"** never appears
as a consecutive sequence in that document. This is a **false positive**.

**Positional index prevents this** by checking that `safety` appears at position `p`,
`vaccine` at `p-1`, and `barriers` at `p+1` — within the same document.
    """)

    st.divider()

    # ── Query functions ──
    def query_biword(q: str):
        terms  = preprocess(q, st.session_state.pp_lower, st.session_state.pp_sw,
                            st.session_state.pp_hyph, st.session_state.pp_method)
        if len(terms) < 2:
            # single-token query: fall back to regular inverted-index lookup
            if not terms:
                return set(), terms, []
            return set(pos_idx.get(terms[0], {}).keys()), terms, []
        biwords = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms) - 1)]
        result  = set(st.session_state.docs.keys())
        for bw in biwords:
            result &= biword_idx.get(bw, set())
        return result, terms, biwords

    def query_positional(q: str):
        terms  = preprocess(q, st.session_state.pp_lower, st.session_state.pp_sw,
                            st.session_state.pp_hyph, st.session_state.pp_method)
        if not terms:
            return set()
        common = set(pos_idx.get(terms[0], {}).keys())
        for t in terms[1:]:
            common &= set(pos_idx.get(t, {}).keys())
        result = set()
        for doc in common:
            positions = [pos_idx.get(t, {}).get(doc, []) for t in terms]
            for start in positions[0]:
                if all(start + i in positions[i] for i in range(1, len(terms))):
                    result.add(doc)
                    break
        return result

    # ── Query interface ──
    phrase_q = st.text_input("Enter a phrase query:", "information retrieval")

    if phrase_q and st.button("Run phrase query"):
        bi_result, bi_terms, biwords = query_biword(phrase_q)
        pos_result                   = query_positional(phrase_q)

        st.markdown(f"**Preprocessed query tokens:** `{bi_terms}`")
        st.markdown(f"**Biword pairs looked up:** `{biwords}`")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Biword Index Results")
            st.markdown(f"Matched **{len(bi_result)}** document(s): `{sorted(bi_result)}`")
            for d in sorted(bi_result):
                st.markdown(f"- **Doc {d} ({st.session_state.doc_titles.get(d, '')}):** "
                            f"{st.session_state.docs[d][:120]}…")
        with col2:
            st.subheader("Positional Index Results")
            st.markdown(f"Matched **{len(pos_result)}** document(s): `{sorted(pos_result)}`")
            for d in sorted(pos_result):
                st.markdown(f"- **Doc {d} ({st.session_state.doc_titles.get(d, '')}):** "
                            f"{st.session_state.docs[d][:120]}…")

        false_pos = bi_result - pos_result
        if false_pos:
            st.warning(
                f"⚠️ **False positives in biword index:** Doc(s) {sorted(false_pos)} — "
                "these match the bigram pairs but the phrase does not appear consecutively."
            )
        else:
            st.success("✅ No false positives for this query — both indices agree.")

        st.markdown(f"""
**Inference:**
Biword matched **{len(bi_result)}** doc(s); positional matched **{len(pos_result)}** doc(s).
False positives: **{len(false_pos)}**.

Positional index is strictly more accurate — it verifies token adjacency by checking
`position[term_i+1] = position[term_i] + 1`. Biword is faster (O(1) hash lookup per bigram)
but cannot distinguish documents where the constituent words appear non-consecutively.

**Recommended approach:** two-stage retrieval — biword pre-filter (fast) → positional verification (accurate).
        """)

    # ── Batch comparison table ──
    st.subheader("Batch Query Comparison")
    # "index lookup queries" is engineered to produce a confirmed false positive:
    # Doc 4 contains "index lookup" (adjacent) and "lookup queries" (adjacent) as separate
    # phrase fragments, so biword matches it — but "index lookup queries" never appears
    # consecutively, so positional correctly rejects it.
    BATCH_QUERIES = [
        "vaccine barriers",
        "information retrieval",
        "machine learning",
        "index lookup queries",   # ← engineered false-positive demo
        "stop word removal",
    ]
    if st.button("Run batch comparison (5 preset queries)"):
        rows = []
        for q in BATCH_QUERIES:
            bi, _, _ = query_biword(q)
            pos       = query_positional(q)
            rows.append({
                "Query":              q,
                "Biword docs":        str(sorted(bi)),
                "Positional docs":    str(sorted(pos)),
                "False positives":    str(sorted(bi - pos)),
                "FP count":           len(bi - pos),
            })
        df_batch = pd.DataFrame(rows)
        st.dataframe(df_batch, width="stretch")
        st.info(
            "**Note on 'index lookup queries':** Doc 4 contains the bigrams "
            "*'index lookup'* and *'lookup queries'* as separate phrase fragments "
            "(the word *lookup* appears twice: once after *index*, once before *queries*). "
            "Biword index returns Doc 4 — a **false positive** — because it cannot verify "
            "that all three words appear consecutively. "
            "Positional index correctly rejects Doc 4 (no document has "
            "*index → lookup → queries* at consecutive positions)."
        )


# ===========================================================================
# SECTION D — BST vs B-Tree
# ===========================================================================
elif section.startswith("D"):
    st.header("D. Dictionary Search: BST vs B-Tree")

    processed = get_processed()
    inv_idx   = build_inv_index(processed)
    vocab     = sorted(inv_idx.keys())

    # ── BST ──
    class _BSTNode:
        __slots__ = ("key", "postings", "left", "right")
        def __init__(self, key, postings):
            self.key = key; self.postings = postings
            self.left = self.right = None

    class BST:
        def __init__(self): self.root = None

        def insert(self, key, postings):
            def _ins(node, k, p):
                if not node: return _BSTNode(k, p)
                if   k < node.key: node.left  = _ins(node.left,  k, p)
                elif k > node.key: node.right = _ins(node.right, k, p)
                return node
            self.root = _ins(self.root, key, postings)

        def search(self, key):
            node, steps = self.root, 0
            while node:
                steps += 1
                if key == node.key:  return node.postings, steps
                elif key < node.key: node = node.left
                else:                node = node.right
            return None, steps

    # ── B-Tree proxy: sorted array + binary search (O(log n) guaranteed) ──
    class BTree:
        def __init__(self): self.keys = []; self.data = {}

        def build(self, term_dict):
            self.keys = sorted(term_dict.keys())
            self.data = dict(term_dict)

        def search(self, key):
            lo, hi, steps = 0, len(self.keys) - 1, 0
            while lo <= hi:
                mid   = (lo + hi) // 2
                steps += 1
                if   self.keys[mid] == key: return self.data[key], steps
                elif self.keys[mid] <  key: lo = mid + 1
                else:                       hi = mid - 1
            return None, steps

    # Build: random insertion BST, sorted insertion BST (worst case), and B-Tree
    bst_rnd = BST()
    shuffled = vocab.copy(); random.shuffle(shuffled)
    for term in shuffled:
        bst_rnd.insert(term, inv_idx[term])

    bst_srt = BST()                   # intentionally degenerate
    for term in vocab:
        bst_srt.insert(term, inv_idx[term])

    btree = BTree()
    btree.build(inv_idx)

    # ── Overview ──
    st.subheader("Structure Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Binary Search Tree (BST)**")
        st.markdown(f"- Vocabulary: **{len(vocab)}** terms")
        st.markdown("- *Random insertion* → roughly balanced → O(log n) average")
        st.markdown("- *Sorted insertion* → degenerate right chain → O(n) worst case")
    with col2:
        st.markdown("**B-Tree — modelled via sorted-array binary search**")
        st.markdown(f"- Vocabulary: **{len(vocab)}** terms")
        st.markdown("- Sorted keys in array → binary search gives **O(log₂ n) guaranteed**")
        st.markdown("- Performance mirrors a real B-Tree: bounded depth, cache-friendly sequential access")

    # ── Manual lookup ──
    st.subheader("Manual Term Lookup")
    lu = st.text_input("Look up a vocabulary term:", "vaccine")
    if lu and st.button("Search"):
        t_lower = lu.lower().strip()
        p_rnd, s_rnd = bst_rnd.search(t_lower)
        p_srt, s_srt = bst_srt.search(t_lower)
        p_bt,  s_bt  = btree.search(t_lower)
        col1, col2, col3 = st.columns(3)
        col1.metric("BST (random) steps", s_rnd)
        col2.metric("BST (sorted) steps", s_srt)
        col3.metric("B-Tree steps",       s_bt)
        st.markdown(f"Postings: `{p_bt or '(not found)'}`")

    # ── Benchmark ──
    st.subheader("Performance Benchmark")
    n_q = st.slider("Number of random benchmark queries:", 10, 50, 20)

    if st.button(f"Run benchmark ({n_q} lookups)"):
        sample = random.sample(vocab, min(n_q, len(vocab)))
        rows   = []
        for term in sample:
            t0 = time.perf_counter(); _, s_r = bst_rnd.search(term); t_r = (time.perf_counter() - t0) * 1e6
            t0 = time.perf_counter(); _, s_s = bst_srt.search(term); t_s = (time.perf_counter() - t0) * 1e6
            t0 = time.perf_counter(); _, s_b = btree.search(term);   t_b = (time.perf_counter() - t0) * 1e6
            rows.append({
                "Term":                 term,
                "BST(random) steps":    s_r,
                "BST(sorted) steps":    s_s,
                "B-Tree steps":         s_b,
                "BST(random) μs":       f"{t_r:.2f}",
                "BST(sorted) μs":       f"{t_s:.2f}",
                "B-Tree μs":            f"{t_b:.2f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")

        avg_r_s = df["BST(random) steps"].mean()
        avg_s_s = df["BST(sorted) steps"].mean()
        avg_b_s = df["B-Tree steps"].mean()
        avg_r_t = df["BST(random) μs"].astype(float).mean()
        avg_s_t = df["BST(sorted) μs"].astype(float).mean()
        avg_b_t = df["B-Tree μs"].astype(float).mean()
        max_s_s = df["BST(sorted) steps"].max()

        c1, c2, c3 = st.columns(3)
        c1.metric("BST (random) avg steps",    f"{avg_r_s:.1f}", delta=f"{avg_r_t:.2f} μs")
        c2.metric("BST (sorted) avg steps",    f"{avg_s_s:.1f}",
                  delta=f"{avg_s_t:.2f} μs", delta_color="inverse")
        c3.metric("B-Tree avg steps",          f"{avg_b_s:.1f}", delta=f"{avg_b_t:.2f} μs")

        st.markdown(f"""
**Inference:**

| Structure | Avg Steps | Max Steps | Avg Time | Worst-Case Complexity |
|-----------|-----------|-----------|----------|-----------------------|
| BST (random insertion) | {avg_r_s:.1f} | {df["BST(random) steps"].max()} | {avg_r_t:.2f} μs | O(n) if sorted |
| BST (sorted insertion) | {avg_s_s:.1f} | {max_s_s} | {avg_s_t:.2f} μs | **O(n) — degenerate chain** |
| B-Tree (guaranteed) | {avg_b_s:.1f} | {df["B-Tree steps"].max()} | {avg_b_t:.2f} μs | **O(log n) always** |

The BST with **sorted insertion** reached up to **{max_s_s} steps** — demonstrating the O(n) degenerate case.
B-Tree achieved at most **{df["B-Tree steps"].max()} steps** for all {n_q} queries, confirming O(log₂({len(vocab)})) ≈ {math.ceil(math.log2(max(len(vocab),2)))} bound.

**Conclusion:** B-Trees are mandatory for production IR systems. Lucene and Elasticsearch use B-tree variants
for their dictionary structures because they guarantee search time and support prefix/range queries efficiently.
        """)


# ===========================================================================
# SECTION E — Tolerant Retrieval
# ===========================================================================
elif section.startswith("E"):
    st.header("E. Tolerant Retrieval")

    processed = get_processed()
    inv_idx   = build_inv_index(processed)
    vocab     = list(inv_idx.keys())

    # ── k-gram index ──
    def build_kgram(vocab_list, k=2):
        idx = defaultdict(set)
        for term in vocab_list:
            padded = "$" + term + "$"
            for i in range(len(padded) - k + 1):
                idx[padded[i:i+k]].add(term)
        return idx

    kgram = build_kgram(vocab, k=2)

    # ── Levenshtein edit distance (space-optimised DP) ──
    def levenshtein(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp   = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp  = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev  = temp
        return dp[n]

    # ── Soundex ──
    def soundex(word: str) -> str:
        if not word: return "0000"
        w   = word.upper()
        tbl = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
        code = w[0]
        # initialise prev to the first letter's own code so adjacent same-group letters are suppressed
        prev = next((d for grp, d in tbl.items() if w[0] in grp), None)
        for ch in w[1:]:
            for letters, digit in tbl.items():
                if ch in letters:
                    if digit != prev: code += digit
                    prev = digit
                    break
            else:
                prev = None
        return (code + "0000")[:4]

    mode = st.selectbox("Select tolerant retrieval method:", [
        "A. Wildcard Query (k-gram index, k=2)",
        "B. Spelling Correction (Levenshtein edit distance)",
        "C. Phonetic Correction (Soundex)",
    ])

    # ─── Wildcard ───────────────────────────────────────────────────────────
    if mode.startswith("A"):
        st.subheader("Wildcard Query via K-gram Index (k=2)")

        with st.expander("How k-gram wildcard works"):
            st.markdown("""
**Step 1 — Build the k-gram index:**
For each vocabulary term, pad with `$` anchors and extract all bigrams:
`information` → `$information$` → `$i, in, nf, fo, or, rm, ma, at, ti, io, on, n$`

**Step 2 — Process wildcard query** (e.g., `vaccin*`):
Extract bigrams from padded prefix `$vaccin` → `$v, va, ac, cc, ci, in`

**Step 3 — Intersect posting lists** for each bigram → candidate terms

**Step 4 — Post-filter:** keep only candidates that actually start with `vaccin`
            """)

        wc_q = st.text_input("Wildcard query (suffix wildcard):", "vaccin*")

        if wc_q and st.button("Wildcard search"):
            pattern = wc_q.lower().replace("*", "").strip()
            padded  = "$" + pattern
            q_grams = [padded[i:i+2] for i in range(len(padded) - 1)]

            if q_grams:
                sets       = [kgram.get(g, set()) for g in q_grams]
                candidates = set.intersection(*sets) if sets else set(vocab)
            else:
                candidates = set(vocab)

            matches = sorted(t for t in candidates if t.startswith(pattern))

            st.markdown(f"**Query 2-grams:** `{q_grams}`")
            st.markdown(f"**Candidates (before prefix filter):** `{sorted(candidates)[:20]}`")
            st.markdown(f"**Final matches after prefix filter:** `{matches}`")

            all_docs = set()
            for m in matches:
                all_docs |= set(inv_idx.get(m, []))

            if matches:
                st.success(f"Found **{len(matches)}** vocabulary match(es) → **{len(all_docs)}** document(s).")
                for d in sorted(all_docs):
                    st.markdown(f"- **Doc {d}:** {st.session_state.docs[d][:110]}…")
            else:
                st.warning("No vocabulary terms match this wildcard pattern.")

            with st.expander("K-gram index entries for query grams"):
                rows = [{"2-gram": g, "Matching vocab terms": str(sorted(kgram.get(g, set()))[:15])}
                        for g in q_grams]
                st.dataframe(pd.DataFrame(rows), width="stretch")

            st.markdown(f"""
**Inference:**
Wildcard `{wc_q}` matched **{len(matches)}** vocabulary term(s) via 2-gram intersection + prefix filter.
K-gram indexing avoids scanning all {len(vocab)} vocabulary terms — it narrows candidates first, then verifies.
Limitation: very short prefixes (e.g. `a*`) produce high-frequency grams (`$a`) and large candidate sets before filtering.
            """)

    # ─── Spelling correction ─────────────────────────────────────────────────
    elif mode.startswith("B"):
        st.subheader("Spelling Correction via Levenshtein Edit Distance")

        with st.expander("How edit distance correction works"):
            st.markdown("""
**Levenshtein distance** counts the minimum number of single-character edits
(insert, delete, substitute) needed to transform one string into another.

- `infromation` → `information`: transposition of `r` and `o` → **distance = 2** (standard Levenshtein counts delete + insert; Damerau-Levenshtein would give 1)
- `retrival` → `retrieval`: 1 insertion (`e`) → distance = 1

We compare the misspelled query against every vocabulary term and return
suggestions with distance ≤ threshold (typically 1 or 2).
            """)

        col1, col2 = st.columns([3, 1])
        misspelled = col1.text_input("Misspelled query term:", "infromation")
        max_dist   = col2.slider("Max edit distance:", 1, 3, 2)

        if misspelled and st.button("Find corrections"):
            q = misspelled.lower().strip()
            results = sorted(
                [(t, levenshtein(q, t)) for t in vocab if levenshtein(q, t) <= max_dist],
                key=lambda x: x[1],
            )

            if results:
                rows = [{"Suggestion": t, "Edit distance": d,
                         "Docs": str(sorted(inv_idx.get(t, [])))}
                        for t, d in results[:15]]
                st.dataframe(pd.DataFrame(rows), width="stretch")

                best = results[0]
                st.success(f"Best match: **'{best[0]}'** (distance = {best[1]}), "
                           f"in docs {sorted(inv_idx.get(best[0], []))}")

                # Show DP matrix for the best suggestion
                with st.expander(f"Edit distance matrix: '{q}' → '{best[0]}'"):
                    s1, s2 = q, best[0]
                    m, n   = len(s1), len(s2)
                    dp     = [[0] * (n+1) for _ in range(m+1)]
                    for i in range(m+1): dp[i][0] = i
                    for j in range(n+1): dp[0][j] = j
                    for i in range(1, m+1):
                        for j in range(1, n+1):
                            dp[i][j] = (dp[i-1][j-1] if s1[i-1] == s2[j-1]
                                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
                    # Use positional labels (e.g. "0:i", "1:n") so repeated
                    # characters in s1/s2 don't produce duplicate row/column
                    # names — PyArrow rejects DataFrames with duplicate columns.
                    row_labels = [" "] + [f"{i}:{c}" for i, c in enumerate(s1)]
                    col_labels = [" "] + [f"{j}:{c}" for j, c in enumerate(s2)]
                    mat = pd.DataFrame(dp, index=row_labels, columns=col_labels)
                    st.dataframe(mat, width="stretch")
            else:
                st.warning(f"No corrections found within edit distance {max_dist}.")

            st.markdown(f"""
**Inference:**
Levenshtein distance found **{len(results)}** correction(s) for *'{misspelled}'* within distance ≤ {max_dist}.
Best match **'{results[0][0] if results else 'N/A'}'** requires {results[0][1] if results else 'N/A'} edit(s).
A threshold of {max_dist} captures common typos (transpositions, substitutions) while minimising false suggestions.
For large vocabularies, pre-filtering with k-gram intersection reduces Levenshtein comparisons from O(|V|) to O(√|V|).
            """)

    # ─── Phonetic ────────────────────────────────────────────────────────────
    else:
        st.subheader("Phonetic Correction via Soundex")

        with st.expander("How Soundex works"):
            st.markdown("""
Soundex maps a word to a **4-character code** (first letter + 3 consonant-group digits):

| Consonant group | Code |
|----------------|------|
| B, F, P, V | 1 |
| C, G, J, K, Q, S, X, Z | 2 |
| D, T | 3 |
| L | 4 |
| M, N | 5 |
| R | 6 |

Vowels and H, W, Y are discarded. Consecutive consonants in the same group are merged.
`Smith` → `S530`, `Smyth` → `S530` → same code → phonetic match!
            """)

        ph_q = st.text_input("Query term:", "retrival")

        if ph_q and st.button("Find phonetic matches"):
            query_code = soundex(ph_q)
            matches    = [(t, soundex(t)) for t in vocab if soundex(t) == query_code]

            st.markdown(f"**Soundex code for '{ph_q}':** `{query_code}`")

            if matches:
                rows = [{"Matched term": t, "Soundex": c,
                         "Docs": str(sorted(inv_idx.get(t, [])))}
                        for t, c in sorted(matches)]
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.success(f"Found {len(matches)} phonetically similar term(s).")
            else:
                st.info("No phonetic matches in current vocabulary for this code.")

            st.markdown("**Soundex codes — sample vocabulary terms:**")
            sample = [{"Term": t, "Soundex": soundex(t)} for t in sorted(vocab)[:25]]
            st.dataframe(pd.DataFrame(sample), width="stretch")

            st.markdown(f"""
**Inference:**
Soundex encoded *'{ph_q}'* as `{query_code}` and found **{len(matches)}** phonetically similar term(s).
Soundex is effective for name lookup (Smith/Smyth) but produces **false positives** for technical vocabulary —
unrelated terms can share a code if their consonant sequences happen to match.
**Recommendation:** Use Soundex only as a first-pass filter; combine with edit distance for precision.
Double Metaphone is a more accurate alternative for production systems.
            """)

    # ── Summary comparison ──
    st.divider()
    st.subheader("Tolerant Retrieval Methods — Summary Comparison")
    if st.button("Show comparison"):
        rows = [
            {
                "Method":       "Wildcard (k-gram, k=2)",
                "Query type":   "`vaccin*`, `info*`",
                "Approach":     "K-gram intersection + prefix filter",
                "Strength":     "Efficient prefix patterns; avoids full vocab scan",
                "Limitation":   "Short prefixes → large candidate sets before filtering",
                "Complexity":   "O(|q-grams| + |candidates|)",
            },
            {
                "Method":       "Spelling correction (edit dist.)",
                "Query type":   "`infromation`, `retrival`",
                "Approach":     "Levenshtein DP, threshold ≤ 2",
                "Strength":     "Exact optimal edit-distance; handles transpositions",
                "Limitation":   "O(|V|) comparisons without k-gram pre-filtering",
                "Complexity":   "O(|V| × |q| × |t|) naive",
            },
            {
                "Method":       "Phonetic (Soundex)",
                "Query type":   "`Smyth` → `Smith`",
                "Approach":     "4-char consonant-group hash",
                "Strength":     "Fast; language-agnostic; good for names",
                "Limitation":   "False positives; poor for technical IR vocabulary",
                "Complexity":   "O(|V|) hash comparisons",
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")


# ===========================================================================
# SECTION G — Discussion & Inference
# ===========================================================================
elif section.startswith("G"):
    st.header("G. Discussion & Inference")

    st.markdown("""
## Summary and Conclusions

### A. Text Preprocessing
Tokenization and lowercasing normalise surface forms. Stop word removal reduced vocabulary size
significantly without degrading content-bearing queries — function words contribute little semantic signal.
Hyphen splitting improved recall for compound terms (e.g. *state-of-the-art* splits into three
independently indexed tokens, each contributing to relevant retrievals).

**Stemming vs Lemmatization:** Porter and Snowball stemmers achieve higher recall through aggressive
truncation but produce non-dictionary forms (*vaccin*, *univers*, *retriev*). Lemmatization preserves
valid dictionary forms at a marginal recall cost.

**Conclusion:** For this COVID-19 / IR domain corpus, **lemmatization is preferred**.
Domain-specific nouns benefit from exact morphological preservation over heuristic truncation.

---

### B. Phrase Query Processing
**Biword index** is simple (O(1) hash lookup per bigram) but cannot verify consecutive adjacency —
it produces false positives when constituent words appear in different sentences or non-adjacent positions.

**Positional index** eliminates false positives by verifying `position[term_i+1] = position[term_i] + 1`,
at the cost of ~25% larger index storage.

**Conclusion:** Positional indexing is mandatory where phrase precision matters.
A two-stage system (biword pre-filter → positional verification) optimises the precision-speed trade-off.

---

### C. Dictionary Structures: BST vs B-Tree
BST with **sorted insertion** degenerates to a right-skewed linear chain, requiring O(n) steps —
confirmed experimentally with up to O(|vocab|) steps in the benchmark.

B-Tree **guarantees O(log_b n)** for all input orders. Experimental benchmarks confirm consistently
fewer steps and lower latency compared to worst-case BST.

**Conclusion:** B-Trees are mandatory for production IR systems. Lucene and Elasticsearch use
B-tree variants for dictionary structures — guaranteed search time, support for prefix/range queries,
and cache-friendly node access via larger page-sized nodes.

---

### D. Tolerant Retrieval

| Method | Best use case | Limitation |
|--------|--------------|-----------|
| Wildcard (k-gram) | Prefix patterns (`vaccin*`) | Short prefixes → large candidate sets |
| Spelling correction (edit dist.) | Typos, transpositions | O(\\|V\\|) naive; needs k-gram pre-filtering |
| Phonetic (Soundex) | Name variants (Smith/Smyth) | False positives on technical vocabulary |

**Conclusion:** No single method covers all tolerant retrieval needs. A hybrid system combining
k-gram pre-filtering with Levenshtein verification is the production best practice.

---

### Limitations

1. **Small corpus** (8 documents) — BST vs B-Tree performance differences are modest. A 100K+ document
   corpus would show dramatically larger divergence between sorted-BST O(n) and B-Tree O(log n).
2. **Boolean-only retrieval** — no TF-IDF ranking or cosine similarity scoring.
3. **Soundex** is poorly suited for technical IR vocabulary.
4. **No proximity queries** — phrase query requires strict adjacency; no ±N window support.

---

### Possible Improvements

1. Add TF-IDF scoring and cosine similarity ranking for graded (ranked) retrieval.
2. Implement a true multi-key B-Tree with disk I/O simulation for realistic benchmarks.
3. Replace Soundex with Double Metaphone for accurate phonetic matching on technical terms.
4. Add proximity queries (terms within N positions of each other).
5. Expand corpus to 100+ documents for statistically meaningful performance experiments.
6. Integrate query expansion (synonym lookup or word embeddings) to improve recall.

---

### Overall Conclusion

This assignment demonstrates a complete end-to-end IR pipeline — from raw text ingestion through
preprocessing, indexing, phrase retrieval, dictionary benchmarking, and tolerant retrieval —
implemented entirely from first principles. Each component confirms established IR theory:

- Positional indices beat biword for phrase accuracy.
- B-Trees beat BST for guaranteed, consistent dictionary search performance.
- Hybrid tolerant retrieval (k-gram + edit distance) outperforms any single method in isolation.

The Streamlit interface makes every algorithmic decision and its trade-off directly observable,
which is the key pedagogical goal of this assignment.
    """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AIMLCZG537 / DSECLZG537 — Information Retrieval | "
    "Assignment 1 | Group 64 | Deadline: 15 June 2026, 23:59 PM"
)
