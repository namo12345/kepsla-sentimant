import re
import pandas as pd
import torch
from symspellpy import SymSpell, Verbosity
import pkg_resources
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ====== Setup SymSpell for light spelling correction ======
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def normalize_text_light(text):
    """Lowercase and remove unwanted special characters (keep .,!?)"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def correct_spelling(text):
    """Correct spelling using SymSpell"""
    words = text.split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected)

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load NLPTown model ======
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "cuda" else -1
)

print("✅ NLPTown model loaded")

# ====== Load CSV ======
INPUT_FILE = "reviews.csv"   # <-- make sure this file is in same folder
df = pd.read_csv(INPUT_FILE)

if "HIGHLIGHTED_REVIEW_CONTENT" not in df.columns:
    raise ValueError("CSV must have column 'HIGHLIGHTED_REVIEW_CONTENT'")

texts = df["HIGHLIGHTED_REVIEW_CONTENT"].astype(str).tolist()

# ====== Preprocess ======
texts = [normalize_text_light(t) for t in texts]
texts = [correct_spelling(t) for t in texts]
df["HIGHLIGHTED_REVIEW_CONTENT"] = texts

# ====== Run NLPTown ======
def analyze_review(review):
    if str(review).strip() == "":
        return "EMPTY", 0.0
    result = sentiment_pipeline(str(review), truncation=True)[0]
    stars = int(result["label"].split()[0])  # e.g. "3 stars"
    if stars <= 2:
        label = "NEGATIVE"
    elif stars == 3:
        label = "NEUTRAL"
    else:
        label = "POSITIVE"
    return label, round(float(result["score"]), 4)

df["sentiment"], df["confidence"] = zip(*[analyze_review(t) for t in texts])

# ====== Save output ======
OUTPUT_FILE = "output_nlptown.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Predictions saved to {OUTPUT_FILE}")
