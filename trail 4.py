# === KePSLA.AI — Aspect-first Mapping + Sentiment + Review-level Aggregation ===
import os
import re
import time
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import torch
import spacy
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =================== Config ===================
BASE_PATH = r"C:\Users\venky\Desktop\kepsla.ai\nlp task"
REVIEWS_CSV = os.path.join(BASE_PATH, "200 reviews.csv")
MAPPING_CSV = os.path.join(BASE_PATH, "mapping_org409_full.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "trail 4 output.csv")

BATCH_SIZE = 32
MAX_SNIPPET_LEN = 32
SIM_THRESHOLD = 0.4
TOP_K_ASPECTS = 5
SNIPPET_WINDOW = 12
BACKOFF_MIN = 0.45
OVERLAP_TAU = 0.25
MARGIN_TAU = 0.10

# =================== Device ===================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA GPU not available. Running on CPU.")

# =================== spaCy ===================
nlp = spacy.load("en_core_web_sm")


@Language.component("smart_comma_boundary")
def set_smart_comma_boundaries(doc: Doc):
    for token in doc[:-1]:
        if token.text == ",":
            next_token = doc[token.i + 1]
            if next_token.text and next_token.text[0].isupper():
                doc[token.i + 1].is_sent_start = True
    return doc


@Language.component("custom_segmenter")
def custom_segmenter(doc: Doc) -> Doc:
    contrasters = {"but", "however", "although", "though", "yet", "nevertheless"}
    for tok in doc[:-1]:
        if tok.text == ";":
            doc[tok.i + 1].is_sent_start = True
        if tok.lower_ in contrasters and tok.i + 1 < len(doc):
            doc[tok.i + 1].is_sent_start = True
        if tok.text == "," and tok.i + 2 < len(doc) and doc[tok.i + 1].pos_ == "CCONJ":
            doc[tok.i + 2].is_sent_start = True
    return doc


nlp.add_pipe("smart_comma_boundary", before="parser")
nlp.add_pipe("custom_segmenter", before="parser")

# =================== Load Data ===================
reviews_df = pd.read_csv(REVIEWS_CSV)
mapping_df = pd.read_csv(MAPPING_CSV)

reviews_df = reviews_df[reviews_df["HIGHLIGHTED_REVIEW_CONTENT"].notnull()].copy()
reviews_df.reset_index(drop=True, inplace=True)

has_review_id = "ID" in reviews_df.columns
has_org_id = "ORGANIZATION_ID" in reviews_df.columns

# Mapping lookup
mapping_lookup = {
    str(row["KEYWORD_NAME"]).strip().lower(): {
        "KPI_ID": row["KPI_ID"],
        "KPI_Name": row["KPI_NAME"],
        "Department_ID": row["DEPARTMENT_ID"],
        "Department_Name": row["DEPARTMENT_NAME"]
    }
    for _, row in mapping_df.iterrows()
}
keyword_texts = list(mapping_lookup.keys())

# =================== Embedding Model ===================
model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
keyword_embeddings = model.encode(keyword_texts, convert_to_tensor=True, device=str(device))

# =================== Sentiment Model (fine-tuned) ===================
SENT_MODEL_PATH = r"./sentiment_model_nlptown"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_PATH, use_fast=True)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH).to(device).eval()

# =================== Utils ===================
STOP_ASPECTS = {"it", "this", "that", "thing", "something", "anything", "everything", "place",
                "hotel", "property", "experience", "stay", "time", "day", "night", "room"}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"[\u2705\u2714\u2713\u263A\u2639\u2764\uFE0F\u2605\u2600-\u26FF]+", ". ", t)
    t = re.sub(r"\+{2,}", ". ", t)
    t = re.sub(r"\u2026+", ". ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def dep_snippet_around(span: spacy.tokens.Span, key_text: str) -> str:
    kw = normalize(key_text)
    match = None
    for tok in span:
        if kw in tok.text.lower() or tok.lemma_.lower() == kw:
            match = tok
            break
    if match is None:
        words = span.text.strip().split()
        return " ".join(words[:MAX_SNIPPET_LEN]) + (" ..." if len(words) > MAX_SNIPPET_LEN else "")
    rel_idx = match.i - span.start
    start = max(0, rel_idx - SNIPPET_WINDOW)
    end = min(len(span), rel_idx + SNIPPET_WINDOW + 1)
    window_tokens = [t.text for t in span[start:end]]
    if len(window_tokens) > MAX_SNIPPET_LEN:
        window_tokens = window_tokens[:MAX_SNIPPET_LEN]
    return " ".join(window_tokens).strip()


def is_informative_chunk(chunk: spacy.tokens.Span) -> bool:
    undesired_ents = {"PERSON", "DATE", "TIME", "QUANTITY", "CARDINAL", "ORDINAL"}
    if any((t.ent_type_ in undesired_ents) for t in chunk):
        return False
    if all(t.pos_ in {"PRON", "DET"} for t in chunk):
        return False
    alpha_tokens = [t for t in chunk if t.is_alpha]
    if len(alpha_tokens) == 1 and alpha_tokens[0].pos_ == "PROPN":
        return False
    if len(alpha_tokens) < 2:
        return False
    if all(not t.is_alpha for t in chunk):
        return False
    if normalize(chunk.text) in STOP_ASPECTS:
        return False
    return True


def extract_semantic_aspects(review: str) -> List[Tuple[str, str]]:
    if not isinstance(review, str) or not review.strip():
        return []
    doc = nlp(preprocess_text(review))
    pairs: List[Tuple[str, str]] = []
    for sent in doc.sents:
        seen_phrases = set()
        for chunk in sent.noun_chunks:
            if not is_informative_chunk(chunk):
                continue
            phrase = normalize(chunk.text)
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            snip = dep_snippet_around(sent, phrase)
            pairs.append((phrase, snip))
    unique = []
    seen = set()
    for a, s in pairs:
        if (a, s) not in seen:
            seen.add((a, s))
            unique.append((a, s))
    return unique


# =================== Sentiment ===================
@torch.inference_mode()
def predict_sentiment_batched(texts: List[str]) -> List[Tuple[str, float, int]]:
    out: List[Tuple[str, float, int]] = []
    if not texts:
        return out
    id2label = {}
    try:
        id2label = getattr(sent_model.config, "id2label", {}) or {}
        id2label = {int(k): str(v) for k, v in (id2label.items() if isinstance(id2label, dict) else [])}
    except Exception:
        id2label = {}
    default_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def normalize_to_triple(pred_index: int, raw_label: str) -> str:
        txt = (raw_label or "").strip().lower()
        if "neg" in txt: return "Negative"
        if "neu" in txt: return "Neutral"
        if "pos" in txt: return "Positive"
        return default_label_map.get(pred_index, "Neutral")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = sent_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        logits = sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()
        for p in probs:
            pred_id = int(p.argmax())
            conf = float(p.max())
            raw = id2label.get(pred_id, default_label_map.get(pred_id, str(pred_id)))
            lab = normalize_to_triple(pred_id, raw)
            stars_map = {"negative": 1, "neutral": 2, "positive": 3}
            star = stars_map.get(lab.lower(), pred_id + 1)
            out.append((lab, conf, star))
        del enc, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# =================== Aggregation ===================
def aggregate_review_sentiment(aspects: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not aspects:
        return "Neutral", 0.0
    sorted_aspects = sorted(aspects, key=lambda a: float(a.get("Confidence_Sentiment", 0.0)), reverse=True)[
                     :TOP_K_ASPECTS]
    weights: Dict[str, float] = defaultdict(float)
    for a in sorted_aspects:
        lab = str(a.get("Aspect_Sentiment", "Neutral"))
        weights[lab] += float(a.get("Confidence_Sentiment", 0.0))
    if not weights:
        return "Neutral", 0.0
    winner, win_weight = max(weights.items(), key=lambda kv: kv[1])
    total_weight = sum(weights.values())
    conf = (win_weight / total_weight) if total_weight > 0 else 0.0
    return winner, conf


# =================== Main ===================
def main():
    t0 = time.time()
    aspect_records_meta: List[Dict[str, Any]] = []
    all_aspect_phrases: List[str] = []
    all_aspect_snippets: List[str] = []

    # --- Extract aspects ---
    for _, row in reviews_df.iterrows():
        review_text = str(row.get("HIGHLIGHTED_REVIEW_CONTENT", "") or "").strip()
        if not review_text:
            continue
        aspects = extract_semantic_aspects(review_text)
        if not aspects:
            aspects = [("general", review_text)]
        for (aspect_phrase, snippet) in aspects:
            rec = {
                "Review_Text": review_text,
                "Aspect_Phrase": aspect_phrase,
                "Aspect_Snippet": snippet,
                "REVIEW_ID": row["ID"] if has_review_id else None,
                "ORGANIZATION_ID": row["ORGANIZATION_ID"] if has_org_id else None
            }
            aspect_records_meta.append(rec)
            all_aspect_phrases.append(aspect_phrase)
            all_aspect_snippets.append(snippet)

    # --- Keyword Mapping ---
    idx = 0
    for b in range(0, len(all_aspect_phrases), BATCH_SIZE):
        batch_phrases = all_aspect_phrases[b:b + BATCH_SIZE]
        batch_embeds = model.encode(batch_phrases, convert_to_tensor=True, device=str(device), batch_size=BATCH_SIZE)
        results = util.semantic_search(batch_embeds, keyword_embeddings, top_k=1)
        for res in results:
            rec = aspect_records_meta[idx]
            if not res:
                rec["Matched_Keyword"] = "Unmapped"
                rec["Cosine_Similarity"] = 0.0
                rec["KPI_ID"] = "Not Found"
                rec["KPI_Name"] = "Not Found"
                rec["Department_ID"] = "Not Found"
                rec["Department_Name"] = "Not Found"
            else:
                best = res[0]
                max_idx = int(best.get('corpus_id', -1))
                max_sim_val = float(best.get('score', 0.0))
                if max_sim_val < SIM_THRESHOLD or max_idx < 0:
                    rec["Matched_Keyword"] = "Unmapped"
                    rec["Cosine_Similarity"] = round(max_sim_val, 4)
                    rec["KPI_ID"] = "Not Found"
                    rec["KPI_Name"] = "Not Found"
                    rec["Department_ID"] = "Not Found"
                    rec["Department_Name"] = "Not Found"
                else:
                    matched_kw = keyword_texts[max_idx]
                    kwd = mapping_lookup.get(matched_kw, {})
                    rec["Matched_Keyword"] = matched_kw
                    rec["Cosine_Similarity"] = round(max_sim_val, 4)
                    rec["KPI_ID"] = kwd.get("KPI_ID", "Not Found")
                    rec["KPI_Name"] = kwd.get("KPI_Name", "Not Found")
                    rec["Department_ID"] = kwd.get("Department_ID", "Not Found")
                    rec["Department_Name"] = kwd.get("Department_Name", "Not Found")
            idx += 1

    # --- Count unmapped BEFORE replacement ---
    unmapped_before = sum(1 for r in aspect_records_meta if str(r.get("Matched_Keyword", "")).lower() == "unmapped")

    # --- Replace Unmapped with N-grams ---
    for r in aspect_records_meta:
        if str(r.get("Matched_Keyword", "")).lower() == "unmapped":
            r["Matched_Keyword"] = r["Aspect_Phrase"]
            r["KPI_ID"] = "N/A"
            r["KPI_Name"] = "N/A"
            r["Department_ID"] = "N/A"
            r["Department_Name"] = "N/A"

    # --- Count unmapped AFTER replacement ---
    unmapped_after = sum(1 for r in aspect_records_meta if str(r.get("Matched_Keyword", "")).lower() == "unmapped")

    # --- Sentiment per aspect ---
    sent_preds = predict_sentiment_batched(all_aspect_snippets)
    final_aspect_rows: List[Dict[str, Any]] = []
    for rec, (lab, conf, stars) in zip(aspect_records_meta, sent_preds):
        out = {
            "REVIEW_ID": rec["REVIEW_ID"],
            "ORGANIZATION_ID": rec["ORGANIZATION_ID"],
            "Original_Review": rec["Review_Text"],
            "N_gram": rec["Aspect_Phrase"],
            "Matched_Keyword": rec["Matched_Keyword"],
            "Cosine_Similarity": rec.get("Cosine_Similarity", 0.0),
            "KPI_ID": rec.get("KPI_ID", "N/A"),
            "KPI_Name": rec.get("KPI_Name", "N/A"),
            "Department_ID": rec.get("Department_ID", "N/A"),
            "Department_Name": rec.get("Department_Name", "N/A"),
            "Aspect_Phrase": rec["Aspect_Phrase"],
            "Aspect_Snippet": rec["Aspect_Snippet"],
            "Aspect_Sentiment": lab,
            "Confidence_Sentiment": round(float(conf), 3),
            "Stars": int(stars),
            "Sentiment_Source": "FineTuned-NLPTown"
        }
        final_aspect_rows.append(out)

    # --- Aggregate review-level ---
    grouped = defaultdict(list)
    for r in final_aspect_rows:
        key = (r["REVIEW_ID"], r["Original_Review"]) if has_review_id else r["Original_Review"]
        grouped[key].append(r)

    for key, rows in grouped.items():
        sentiment, conf = aggregate_review_sentiment(rows)
        for r in rows:
            r["Review_Sentiment"] = sentiment
            r["Review_Confidence"] = round(conf, 3)
            r["Aggregation_Method"] = "MajorityVote+TieBreak+NeutralFix"

    # --- Save final file ---
    aspect_df = pd.DataFrame(final_aspect_rows)
    aspect_df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\n✅ Done: {len(reviews_df)} reviews → {len(aspect_df)} aspect rows.")
    print(f"📂 Final Output: {OUTPUT_CSV}")
    print(f"❌ Unmapped BEFORE replacement: {unmapped_before}")
    print(f"✅ Unmapped AFTER replacement: {unmapped_after}")
    print(f"🚀 Batch size: {BATCH_SIZE}, Device: {device}")
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
