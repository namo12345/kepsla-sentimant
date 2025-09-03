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
BASE_PATH          = r"C:\Users\venky\Desktop\kepsla.ai\nlp task"
REVIEWS_CSV        = os.path.join(BASE_PATH, "reviews.csv")
MAPPING_CSV        = os.path.join(BASE_PATH, "mapping_org409_full.csv")
OUTPUT_CSV         = os.path.join(BASE_PATH, "final_output22.csv")

BATCH_SIZE         = 32
MAX_SNIPPET_LEN    = 48

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
            doc[tok.i+1].is_sent_start = True
        if tok.lower_ in contrasters and tok.i + 1 < len(doc):
            doc[tok.i+1].is_sent_start = True
        if tok.text == "," and tok.i + 2 < len(doc) and doc[tok.i+1].pos_ == "CCONJ":
            doc[tok.i+2].is_sent_start = True
    return doc

nlp.add_pipe("smart_comma_boundary", before="parser")
nlp.add_pipe("custom_segmenter", before="parser")

# =================== Load Data ===================
reviews_df = pd.read_csv(REVIEWS_CSV)
mapping_df = pd.read_csv(MAPPING_CSV)

reviews_df = reviews_df[reviews_df["HIGHLIGHTED_REVIEW_CONTENT"].notnull()].copy()
reviews_df.reset_index(drop=True, inplace=True)

has_review_id = "ID" in reviews_df.columns
has_org_id    = "ORGANIZATION_ID" in reviews_df.columns

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
SENT_MODEL_PATH = r"./sentiment_model_nlptown"   # fine-tuned local model
sent_tokenizer  = AutoTokenizer.from_pretrained(SENT_MODEL_PATH, use_fast=True)
sent_model      = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH).to(device).eval()

# =================== Utils ===================
STOP_ASPECTS = {"it","this","that","thing","something","anything","everything","place","hotel","property","experience","stay","time","day","night","room"}

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def trim_tokens_by_len(tokens: List[spacy.tokens.Token], max_len: int) -> List[spacy.tokens.Token]:
    if len(tokens) <= max_len:
        return tokens
    mid = len(tokens) // 2
    half = max_len // 2
    start = max(0, mid - half)
    end   = min(len(tokens), start + max_len)
    return tokens[start:end]

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
    subtree = list(match.subtree)
    extra = []
    head = match.head
    if head != match and head.pos_ in {"ADJ","VERB","AUX","NOUN"}:
        extra = list(head.subtree)
    merged = []
    seen = set()
    for t in subtree + extra:
        if t.i not in seen:
            merged.append(t)
            seen.add(t.i)
    merged = trim_tokens_by_len(merged, MAX_SNIPPET_LEN)
    merged.sort(key=lambda x: x.i)
    return " ".join(t.text for t in merged).strip()

def extract_semantic_aspects(review: str) -> List[Tuple[str, str]]:
    if not isinstance(review, str) or not review.strip():
        return []
    doc = nlp(review)
    pairs: List[Tuple[str, str]] = []
    for sent in doc.sents:
        seen_phrases = set()
        for chunk in sent.noun_chunks:
            phrase = normalize(chunk.text)
            if not phrase or phrase in STOP_ASPECTS:
                continue
            if len(phrase) <= 2:
                continue
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

@torch.inference_mode()
def predict_sentiment_batched(texts: List[str]) -> List[Tuple[str, float, int]]:
    out: List[Tuple[str, float, int]] = []
    if not texts:
        return out
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = sent_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        logits = sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()
        for p in probs:
            star = int(p.argmax()) + 1
            conf = float(p.max())
            # Map according to fine-tuned labels (0=Negative,1=Neutral,2=Positive)
            if star == 1: lab = "Negative"
            elif star == 2: lab = "Neutral"
            else:           lab = "Positive"
            out.append((lab, conf, star))
        del enc, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out

# =================== Aggregation Logic ===================
def aggregate_review_sentiment(aspects: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not aspects:
        return "Neutral", 0.0

    pos = [a for a in aspects if a["Aspect_Sentiment"] == "Positive"]
    neg = [a for a in aspects if a["Aspect_Sentiment"] == "Negative"]
    neu = [a for a in aspects if a["Aspect_Sentiment"] == "Neutral"]

    total = len(aspects)

    # Neutral inflation fix
    if (len(pos) + len(neg)) >= 0.6 * total:
        counts = {"Positive": len(pos), "Negative": len(neg)}
        max_count = max(counts.values())
        winners = [k for k,v in counts.items() if v == max_count]
    else:
        counts = {"Positive": len(pos), "Negative": len(neg), "Neutral": len(neu)}
        max_count = max(counts.values())
        winners = [k for k,v in counts.items() if v == max_count]

    # Tie-break with avg confidence
    if len(winners) == 1:
        return winners[0], 1.0
    elif set(winners) == {"Positive", "Negative"}:
        avg_conf_pos = sum(a["Confidence_Sentiment"] for a in pos) / len(pos) if pos else 0
        avg_conf_neg = sum(a["Confidence_Sentiment"] for a in neg) / len(neg) if neg else 0
        if avg_conf_pos > avg_conf_neg:
            return "Positive", avg_conf_pos
        elif avg_conf_neg > avg_conf_pos:
            return "Negative", avg_conf_neg
        else:
            return "Neutral", 0.5
    else:
        return "Neutral", 0.5

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

    # --- Keyword Mapping (always best match, no threshold) ---
    aspect_embeds = model.encode(all_aspect_phrases, convert_to_tensor=True, device=str(device))
    cos_mat = util.cos_sim(aspect_embeds, keyword_embeddings)
    for i, rec in enumerate(aspect_records_meta):
        sims = cos_mat[i]
        max_idx     = int(sims.argmax().item())
        max_sim_val = float(sims[max_idx].item())
        matched_kw  = keyword_texts[max_idx]
        kwd = mapping_lookup.get(matched_kw, {})
        rec["Matched_Keyword"] = matched_kw
        rec["Cosine_Similarity"] = round(max_sim_val, 4)
        rec["KPI_ID"] = kwd.get("KPI_ID", "Not Found")
        rec["KPI_Name"] = kwd.get("KPI_Name", "Not Found")
        rec["Department_ID"] = kwd.get("Department_ID", "Not Found")
        rec["Department_Name"] = kwd.get("Department_Name", "Not Found")

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
            "Cosine_Similarity": rec["Cosine_Similarity"],
            "KPI_ID": rec["KPI_ID"],
            "KPI_Name": rec["KPI_Name"],
            "Department_ID": rec["Department_ID"],
            "Department_Name": rec["Department_Name"],
            "Aspect_Phrase": rec["Aspect_Phrase"],
            "Aspect_Snippet": rec["Aspect_Snippet"],
            "Aspect_Sentiment": lab,
            "Confidence_Sentiment": round(float(conf), 3),
            "Stars": int(stars),
            "Sentiment_Source": "FineTuned-NLPTown"
        }
        final_aspect_rows.append(out)

    # --- Aggregate to review-level ---
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
    print(f"🚀 Batch size: {BATCH_SIZE}, Device: {device}")
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
