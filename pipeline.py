# uncompyle6 version 3.9.2
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.7.9 (tags/v3.7.9:13c94747c7, Aug 17 2020, 18:58:18) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: pipeline.py
# Compiled at: 2025-09-29 01:33:32
# Size of source mod 2**32: 82035 bytes
from __future__ import annotations
import argparse, json, logging, math, os, random, itertools
from collections import Counter, defaultdict
import re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np, pandas as pd, nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from langdetect import detect
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, davies_bouldin_score, f1_score, precision_recall_fscore_support, silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    hdbscan = None
    HAS_HDBSCAN = False

try:
    import faiss
except ImportError as exc:
    try:
        raise RuntimeError("faiss-cpu is required for this pipeline") from exc
    finally:
        exc = None
        del exc

def resolve_device() -> "str":
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


@dataclass
class Settings:
    input_csv = "reviews.csv"
    input_csv: "str"
    kpi_metadata_path = "kpi_names.csv"
    kpi_metadata_path: "str"
    department_metadata_path = "department_names.csv"
    department_metadata_path: "str"
    text_column = "HIGHLIGHTED_REVIEW_CONTENT"
    text_column: "str"
    id_column = "ORGANIZATION_ID"
    id_column: "str"
    run_name = field(default_factory=(lambda: f"run-{int(time.time())}"))
    run_name: "str"
    keybert_top_n = 6
    keybert_top_n: "int"
    keybert_min_cosine = 0.6
    keybert_min_cosine: "float"
    dedup_similarity_threshold = 0.9
    dedup_similarity_threshold: "float"
    max_phrase_doc_freq = 0.25
    max_phrase_doc_freq: "float"
    cluster_min_k = 2
    cluster_min_k: "int"
    cluster_max_k = 40
    cluster_max_k: "int"
    silhouette_fallback = 0.2
    silhouette_fallback: "float"
    silhouette_switch_threshold = 0.25
    silhouette_switch_threshold: "float"
    davies_switch_threshold = 1.0
    davies_switch_threshold: "float"
    kpi_threshold = 0.6
    kpi_threshold: "float"
    min_ngram_freq = 2
    min_ngram_freq: "int"
    min_single_word_senses = 2
    min_single_word_senses: "int"
    cluster_min_size = 3
    cluster_min_size: "int"
    cluster_noise_threshold = 0.5
    cluster_noise_threshold: "float"
    cluster_algorithm = "auto"
    cluster_algorithm: "str"
    hdbscan_min_cluster_size = 5
    hdbscan_min_cluster_size: "int"
    hdbscan_min_samples = None
    hdbscan_min_samples: "Optional[int]"
    output_serial_start = 4
    output_serial_start: "int"
    diagnostic_clusters_top_n = 5
    diagnostic_clusters_top_n: "int"
    diagnostic_top_keywords = 10
    diagnostic_top_keywords: "int"
    random_seed = 42
    random_seed: "int"
    output_dir = "outputs"
    output_dir: "str"
    output_file = None
    output_file: "Optional[str]"
    enable_umap = False
    enable_umap: "bool"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_name: "str"
    embedding_fallback_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_fallback_name: "str"
    rule_filter_min_confidence = 0.5
    rule_filter_min_confidence: "float"
    rule_training_confidence = 0.7
    rule_training_confidence: "float"
    rule_override_confidence = 0.7
    rule_override_confidence: "float"


settings = Settings()
random.seed(settings.random_seed)
np.random.seed(settings.random_seed)
logging.basicConfig(level=(logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("keyword_pipeline")
EMBEDDER_CACHE: "Dict[str, SentenceTransformer]" = {}
KEYBERT_CACHE: "Dict[int, KeyBERT]" = {}

def ensure_nltk_resources() -> "None":
    resources = {
     'punkt': '"tokenizers/punkt"', 
     'averaged_perceptron_tagger': '"taggers/averaged_perceptron_tagger"', 
     'wordnet': '"corpora/wordnet"', 
     'omw-1.4': '"corpora/omw-1.4"', 
     'stopwords': '"corpora/stopwords"'}
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()
STOP_WORDS: "Set[str]" = set(stopwords.words("english"))
ALLOWED_POS: "Set[str]" = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
LEMMATIZER = WordNetLemmatizer()
try:
    SPELL_CHECKER = SpellChecker(distance=1)
except Exception:
    SPELL_CHECKER = None

NOISE_NOUNS: "Set[str]" = {
 'item', 'items', 'request', 'requests', 'cleaning', 'tidying', 'tidy', 'thing', 
 'things', 
 'stuff', 'job', 'jobs', 'need', 'needs', 'charge', 'charges', 
 'fee', 
 'fees', 'floor', 'floors', 'person', 'people', 'work', 'effort', 
 'issue', 
 'issues', 'requester', 'requesting', 'management', 'policy', 'policies', 
 'process', 
 'processes', 'case', 'cases', 'matter', 'matters'}
ABSTRACT_IDEA_NOUNS: "Set[str]" = {
 'time', 'love', 'idea', 'happiness', 'joy', 'sadness', 'anger', 'freedom', 
 'justice', 
 'beauty', 'information', 'knowledge', 'energy', 'service', 'experience', 
 'quality', 
 'value', 'emotion', 'feeling', 'truth', 'opinion', 'issue', 
 'problem', 'satisfaction', 
 'comfort', 'management', 'team', 'performance', 
 'feedback'}
PRAISE_TARGETS: "Set[str]" = {
 'job', 'work', 'effort', 'team', 'person', 'people', 'staff', 'manager', 
 'agent', 
 'service', 'support', 'crew'}
ABSTRACT_SUFFIXES: "Tuple[str, ...]" = ('ness', 'ity', 'tion', 'sion', 'ment', 'ance',
                                        'ence', 'hood', 'ship', 'ology')
HOTEL_DOMAIN_NOUNS: "Set[str]" = {
 'hotel', 'room', 'suite', 'apartment', 'service', 'staff', 'food', 'meal', 
 'breakfast', 
 'dinner', 'lunch', 'restaurant', 'bar', 'cafe', 'buffet', 'kitchen', 
 'chef', 
 'menu', 'beverage', 'drink', 'lobby', 'reception', 'frontdesk', 'check', 
 'checkin', 
 'checkout', 'bed', 'bedding', 'mattress', 'pillow', 'linen', 'bathroom', 
 'toilet', 
 'shower', 'toiletries', 'amenity', 'amenities', 'wifi', 'internet', 'network', 
 'connection', 
 'parking', 'valet', 'garage', 'pool', 'spa', 'gym', 'fitness', 
 'sauna', 
 'steam', 'location', 'view', 'beach', 'garden', 'balcony', 'housekeeping', 
 'laundry', 
 'maintenance', 'cleaner', 'security', 'concierge', 'porter', 'bellboy', 
 'transport', 
 'transportation', 'shuttle', 'airport'}
IRRELEVANT_TOKENS: "Set[str]" = {
 'day', 'night', 'time', 'times', 'week', 'month', 'year', 'hour', 
 'minute', 
 'today', 'yesterday', 'tomorrow', 'ago', 'later', 'early', 'late', 
 'first', 
 'second', 'third', 'last', 'next', 'previous', 'every', 'always', 
 'never', 
 'sometimes', 'usually', 'often', 'really', 'quite', 'pretty', 'much', 
 'many', 
 'lot', 'lots', 'bit', 'little', 'big', 'small', 'large', 
 'huge', 'tiny', 
 'place', 'thing', 'things', 'way', 'ways', 'part', 
 'side', 'end', 'beginning', 
 'middle', 'top', 'bottom', 'back', 'front', 
 'left', 'right', 'around', 
 'near', 'far', 'close', 'away', 'sure', 
 'yes', 'ok', 'okay', 'well', 
 'fine', 'alright', 'maybe', 'perhaps', 
 'probably', 'definitely', 'certainly'}
GENERIC_PHRASE_COMPONENTS: "Set[str]" = {
 'hotel', 'stay', 'guest', 'experience', 'place', 'something', 'anything', 
 'everything', 
 'everyone', 'good', 'great', 'nice', 'excellent', 'amazing', 
 'awesome', 'wonderful', 
 'best', 'worst', 'bad', 'poor', 'friendly', 
 'unfriendly', 'comfortable', 'uncomfortable', 
 'fantastic', 'awful', 'terrible', 
 'horrible'}
ASPECT_NOUN_WHITELIST: "Set[str]" = {
 'room', 'staff', 'service', 'housekeeping', 'concierge', 'porter', 'bellboy', 
 'manager', 
 'team', 'breakfast', 'dinner', 'lunch', 'meal', 'buffet', 
 'menu', 'chef', 
 'kitchen', 'taste', 'flavour', 'flavor', 'restaurant', 
 'bar', 'cafe', 'beverage', 
 'drink', 'coffee', 'tea', 'dessert', 'bathroom', 
 'toilet', 'shower', 'toiletries', 
 'towel', 'linen', 'bed', 'bedding', 
 'mattress', 'pillow', 'amenity', 'amenities', 
 'facility', 'facilities', 
 'wifi', 'internet', 'network', 'connection', 'parking', 'valet', 
 'garage', 
 'pool', 'spa', 'gym', 'fitness', 'sauna', 'steam', 'location', 
 'view', 
 'checkout', 'checkin', 'booking', 'reservation', 'value', 'price', 'cost', 
 'speed', 
 'size', 'hygiene', 'cleanliness', 'ambience', 'ambiance', 'music', 'lighting', 
 'aircon', 
 'ac', 'temperature', 'noise', 'security', 'safety', 'comfort', 'transport', 
 'transportation', 
 'shuttle', 'airport'}
FUNCTIONAL_ADJECTIVES: "Set[str]" = {
 'clean', 'dirty', 'noisy', 'quiet', 'fast', 'slow', 'free', 'paid', 
 'available', 
 'crowded', 'spacious', 'comfortable', 'uncomfortable', 'helpful', 'unhelpful', 
 'responsive', 
 'unresponsive', 'efficient', 'inefficient', 'friendly', 
 'unfriendly'}
WORDNET_SENSE_CACHE: "Dict[str, int]" = {}
ABSTRACT_CACHE: "Dict[str, Optional[bool]]" = {}

def noun_synset_count(lemma: "str") -> "int":
    cached = WORDNET_SENSE_CACHE.get(lemma)
    if cached is not None:
        return cached
    synsets = wordnet.synsets(lemma, pos=(wordnet.NOUN))
    count = len(synsets)
    WORDNET_SENSE_CACHE[lemma] = count
    return count


def _wordnet_abstract_signal(lemma: "str") -> "Optional[bool]":
    cached = ABSTRACT_CACHE.get(lemma)
    if cached is not None:
        return cached
    synsets = wordnet.synsets(lemma, pos=(wordnet.NOUN))
    if not synsets:
        ABSTRACT_CACHE[lemma] = None
        return
    abstract_domains = {'noun.cognition', 'noun.feeling', 'noun.state', 'noun.attribute', 'noun.act', 
     'noun.event'}
    concrete_domains = {
     'noun.artifact', 'noun.object', 'noun.food', 'noun.location', 'noun.plant', 
     'noun.animal', 'noun.substance'}
    for syn in synsets:
        if syn.lexname() in concrete_domains:
            ABSTRACT_CACHE[lemma] = False
            return False

    for syn in synsets:
        if syn.lexname() in abstract_domains:
            ABSTRACT_CACHE[lemma] = True
            return True

    ABSTRACT_CACHE[lemma] = None


def is_abstract_lemma(lemma: "str") -> "bool":
    if lemma in ABSTRACT_IDEA_NOUNS:
        return True
    if lemma.endswith(ABSTRACT_SUFFIXES):
        return True
    signal = _wordnet_abstract_signal(lemma)
    if signal is True:
        return True
    if signal is False:
        return False
    return False


def is_concrete_noun(lemma: "str") -> "bool":
    if lemma in HOTEL_DOMAIN_NOUNS:
        return True
    if lemma in NOISE_NOUNS or lemma in ABSTRACT_IDEA_NOUNS:
        return False
    return not is_abstract_lemma(lemma)


def is_generic_phrase(phrase: "str") -> "bool":
    tokens = phrase.split()
    if not tokens:
        return True
    return all((token in GENERIC_PHRASE_COMPONENTS for token in tokens))


def load_domain_metadata(settings: "Settings") -> "Tuple[List[str], List[str]]":
    kpi_names = []
    department_names = []
    kpi_path = Path(settings.kpi_metadata_path)
    if kpi_path.exists():
        df = pd.read_csv(kpi_path, encoding="utf-8-sig").dropna()
        kpi_names = df.iloc[:, 0].astype(str).tolist()
        logger.info("Loaded %s KPI labels from %s", len(kpi_names), kpi_path)
    else:
        logger.warning("KPI metadata file %s not found; falling back to defaults", kpi_path)
        kpi_names = [
         'Activities', 
         'Amenities', 
         'Customer Satisfaction', 
         'Room', 
         'Service', 
         'FoodandBeverages']
    dept_path = Path(settings.department_metadata_path)
    if dept_path.exists():
        df = pd.read_csv(dept_path, encoding="utf-8-sig").dropna()
        department_names = df.iloc[:, 0].astype(str).tolist()
        logger.info("Loaded %s departments from %s", len(department_names), dept_path)
    else:
        logger.warning("Department metadata file %s not found; using defaults", dept_path)
        department_names = [
         'Front Office', 
         'Housekeeping', 
         'Restaurant', 
         'Room', 
         'General']
    return (
     kpi_names, department_names)


def infer_department(kpi_name, tokens, department_names):
    hints = {
     'food': '"Restaurant"', 
     'dining': '"Restaurant"', 
     'bar': '"Bar/ Pub"', 
     'clean': '"Housekeeping"', 
     'hygiene': '"Housekeeping"', 
     'room': '"Room"', 
     'service': '"Front Office"', 
     'staff': '"Front Office"', 
     'security': '"Security"', 
     'safety': '"Safety"', 
     'price': '"Retail"', 
     'value': '"Retail"', 
     'wifi': '"Internet / Wifi"', 
     'pool': '"Recreation"', 
     'location': '"Location"', 
     'spa': '"Spa"', 
     'gym': '"Gym"', 
     'comfort': '"Room"', 
     'smell': '"Housekeeping"', 
     'ambience': '"Restaurant"'}
    tok_list = list(tokens)
    for token in tok_list:
        for hint, department in hints.items():
            if hint in token:
                return department

    normalised = [re.sub("[^a-z]", "", d.lower()) for d in department_names]
    for department, norm in zip(department_names, normalised):
        if not norm:
            continue
        for token in tok_list:
            if norm in re.sub("[^a-z]", "", token):
                return department

    if department_names:
        return department_names[0]
    return "General"


def generate_domain_rules(kpi_names, department_names, settings):
    token_synonyms = {'hygiene':[
      "cleanliness", "sanitation"], 
     'clean':[
      "sanitary", "spotless", "hygienic"], 
     'staff':[
      'personnel', 'team', 'crew', 'associates', 'employees'], 
     'service':[
      "support", "assistance", "hospitality", "customer care"], 
     'food':[
      'cuisine', 'meal', 'dining', 'restaurant', 'kitchen'], 
     'taste':[
      "flavour", "flavor", "palate", "seasoning"], 
     'buffet':[
      "spread", "banquet", "breakfast"], 
     'breakfast':[
      "brunch", "morning meal"], 
     'lunch':[
      "midday meal", "tiffin"], 
     'dinner':[
      "supper", "evening meal"], 
     'beverage':[
      "drink", "refreshment"], 
     'location':[
      "proximity", "area", "neighbourhood", "neighborhood"], 
     'room':[
      "suite", "bedroom", "accommodation", "lodging"], 
     'amenities':[
      "facility", "amenity", "features", "utilities"], 
     'safety':[
      "security", "protection"], 
     'price':[
      "cost", "rate", "tariff", "value"], 
     'comfort':[
      "comfortable", "relaxation"], 
     'speed':[
      "latency", "response", "quickness"], 
     'wifi':[
      "internet", "network", "wi-fi", "broadband"], 
     'pool':[
      "swimming pool", "plunge"], 
     'spa':[
      "wellness", "salon"], 
     'parking':[
      "garage", "valet", "carpark"], 
     'waiter':[
      "server", "steward"], 
     'housekeeping':[
      "cleaning", "maid"], 
     'check':[
      "checkin", "checkout", "registration"], 
     'ambience':[
      "ambiance", "atmosphere", "vibe"], 
     'music':[
      "soundtrack", "playlist"], 
     'lighting':[
      "illumination", "lights"], 
     'noise':[
      "sound", "volume"], 
     'security':[
      "safety", "guard"], 
     'transport':[
      "shuttle", "transfer", "pickup", "dropoff"], 
     'value':[
      "pricing", "worth", "affordability"], 
     'reservation':[
      "booking", "prebooking"]}
    token_sets = []
    df_counts = Counter()
    for raw_kpi in sorted({str(k).strip() for k in kpi_names if str(k).strip()}):
        tokens = [t for t in re.findall("[a-zA-Z]+", raw_kpi.lower()) if t]
        if not tokens:
            continue
        unique_tokens = sorted(set(tokens))
        token_sets.append((raw_kpi, tokens, unique_tokens))
        df_counts.update(unique_tokens)

    total_docs = max(1, len(token_sets))
    rules = []
    for raw_kpi, tokens, unique_tokens in token_sets:
        keyword_weights = {}
        for token in unique_tokens:
            df = df_counts.get(token, 0)
            idf = math.log((1 + total_docs) / (1 + df)) + 1.0
            keyword_weights[token] = idf

        for token in unique_tokens:
            for synonym in token_synonyms.get(token, []):
                syn = synonym.lower()
                keyword_weights.setdefault(syn, 0.45)

        primary_phrase = " ".join(tokens)
        department = infer_department(raw_kpi, unique_tokens, department_names)
        target_weight = sum((keyword_weights.get(token, 1.0) for token in unique_tokens))
        rules.append({'kpi':raw_kpi, 
         'department':department, 
         'keywords':sorted(keyword_weights.keys()), 
         'keyword_weights':keyword_weights, 
         'primary_tokens':unique_tokens, 
         'primary_phrase':primary_phrase, 
         'target_weight':max(target_weight, 1.0)})

    logger.info("Generated %s domain rules", len(rules))
    return rules


def apply_domain_rules(phrase: "str", rules: "List[Dict[str, Any]]") -> "Optional[Dict[str, Any]]":
    phrase_lc = phrase.lower()
    best_match = None
    for rule in rules:
        matched_weight = 0.0
        matched_keywords = []
        for keyword, weight in rule.get("keyword_weights", {}).items():
            if keyword and keyword in phrase_lc:
                matched_weight += weight
                matched_keywords.append(keyword)

        if matched_weight <= 0:
            continue
        target_weight = rule.get("target_weight", 1.0)
        confidence = min(1.0, matched_weight / max(target_weight, 1e-06))
        candidate = {'kpi':rule["kpi"], 
         'department':rule["department"], 
         'confidence':confidence, 
         'matched_keywords':matched_keywords}
        if best_match is None or candidate["confidence"] > best_match["confidence"]:
            best_match = candidate

    return best_match


def annotate_domain_matches(df, rules, min_confidence):
    annotated = []
    for row in df.itertuples():
        match = apply_domain_rules(row.phrase, rules)
        if match:
            if match["confidence"] < min_confidence:
                continue
            record = row._asdict()
            record.pop("Index", None)
            record["domain_kpi"] = match["kpi"]
            record["domain_department"] = match["department"]
            record["domain_confidence"] = match["confidence"]
            annotated.append(record)

    if not annotated:
        logger.warning("Domain rules produced no matches; terminating pipeline")
        extra_cols = ["domain_kpi", "domain_department", "domain_confidence"]
        return pd.DataFrame(columns=(list(df.columns) + extra_cols))
    result = pd.DataFrame(annotated)
    logger.info("Retained %s phrases after domain filtering", len(result))
    return result


def load_embedder(model_name: "str") -> "SentenceTransformer":
    if model_name in EMBEDDER_CACHE:
        return EMBEDDER_CACHE[model_name]
    device = resolve_device()
    logger.info("Loading SentenceTransformer %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    EMBEDDER_CACHE[model_name] = model
    return model


def get_embedder(settings: "Settings") -> "SentenceTransformer":
    primary = settings.embedding_model_name
    fallback = settings.embedding_fallback_name
    if primary == fallback:
        fallback = "sentence-transformers/all-mpnet-base-v2"
    try:
        return load_embedder(primary)
    except Exception as exc:
        try:
            logger.warning("Primary embedder %s failed (%s); falling back to %s", primary, exc, fallback)
            return load_embedder(fallback)
        finally:
            exc = None
            del exc


def encode_phrases(phrases: "List[str]", embedder: "SentenceTransformer") -> "np.ndarray":
    return embedder.encode(phrases,
      batch_size=32,
      convert_to_numpy=True,
      show_progress_bar=False).astype(np.float32)
def get_keybert(embedder: "SentenceTransformer") -> "KeyBERT":
    key = id(embedder)
    if key not in KEYBERT_CACHE:
        KEYBERT_CACHE[key] = KeyBERT(model=embedder)
    return KEYBERT_CACHE[key]




def normalize_text(text: "str") -> "str":
    return " ".join(str(text).replace("\n", " ").split()).strip().lower()


def detect_language_safe(text: "str") -> "str":
    try:
        return detect(text)[:5]
    except Exception:
        return "en"


def ingest_reviews(settings: "Settings") -> "pd.DataFrame":
    path = Path(settings.input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV {path} not found")
    df = pd.read_csv(path)
    if settings.text_column not in df.columns:
        raise KeyError(f"Text column {settings.text_column} is missing from input data")
    text_series = df[settings.text_column].fillna("").astype(str)
    id_series = df[settings.id_column] if settings.id_column in df.columns else pd.Series(range(len(df)))
    records = []
    skipped_language = 0
    skipped_empty = 0
    for raw_id, raw_text in zip(id_series, text_series):
        normalized = normalize_text(raw_text)
        if not normalized:
            skipped_empty += 1
            continue
        if detect_language_safe(normalized) != "en":
            skipped_language += 1
            continue
        lemmas, pos_tags, clean_text = tokenize_and_lemmatize(normalized)
        if not lemmas:
            skipped_empty += 1
            continue
        records.append({
         'source_id': raw_id,
         'raw_text': str(raw_text),
         'normalized_text': normalized,
         'clean_text': clean_text,
         'tokens': lemmas,
         'pos_tags': pos_tags
         })

    result = pd.DataFrame(records)
    logger.info("Ingested %d rows (skipped %d empty, %d non-English)", len(result), skipped_empty, skipped_language)
    if result.empty:
        raise RuntimeError("No usable records after ingestion")
    return result


def _wordnet_pos(tag: "str") -> "str":
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize_and_lemmatize(text: "str") -> "Tuple[List[str], List[str], str]":
    raw_tokens = word_tokenize(text) if text else []
    cleaned_tokens = []
    for token in raw_tokens:
        token_clean = re.sub("[^a-zA-Z]+", "", token).lower()
        if not token_clean:
            continue
        if len(token_clean) <= 2:
            continue
        if not token_clean in STOP_WORDS:
            if token_clean in IRRELEVANT_TOKENS:
                continue
            if SPELL_CHECKER is not None:
                if token_clean in SPELL_CHECKER.unknown([token_clean]):
                    corrected = SPELL_CHECKER.correction(token_clean)
                    if corrected:
                        token_clean = corrected.lower()
            cleaned_tokens.append(token_clean)

    if not cleaned_tokens:
        return ([], [], "")
    pos_tags = pos_tag(cleaned_tokens)
    lemmas = []
    pos_prefixes = []
    for word, tag in pos_tags:
        pos_prefix = tag[:2]
        if pos_prefix not in frozenset({'NN', 'JJ'}):
            continue
        lemma = LEMMATIZER.lemmatize(word, _wordnet_pos(tag))
        if lemma:
            if lemma in STOP_WORDS or len(lemma) <= 2:
                continue
            if pos_prefix == "NN":
                if lemma in NOISE_NOUNS:
                    continue
                if lemma in ABSTRACT_IDEA_NOUNS:
                    continue
                if lemma not in HOTEL_DOMAIN_NOUNS and lemma not in ASPECT_NOUN_WHITELIST:
                    if not is_concrete_noun(lemma):
                        continue
            lemmas.append(lemma)
            pos_prefixes.append(pos_prefix)

    clean_text = " ".join(lemmas) if lemmas else ""
    return (lemmas, pos_prefixes, clean_text)


def extract_keywords(df: "pd.DataFrame", settings: "Settings", embedder: "SentenceTransformer") -> "Tuple[pd.DataFrame, Dict[str, Any]]":
    stats = {
        'documents': max(len(df), 1),
        'raw_unigrams': 0,
        'raw_bigrams': 0,
        'raw_trigrams': 0,
        'raw_candidates': 0,
        'unique_candidates': 0,
        'post_freq_candidates': 0,
        'low_freq_pruned': 0,
        'high_docfreq_pruned': 0,
        'similarity_pruned': 0,
        'token_pruned': 0,
        'wordnet_pruned': 0,
        'sense_pruned': 0,
        'noise_pruned': 0,
        'abstract_pruned': 0,
        'pattern_pruned': 0,
        'generic_pruned': 0,
        'praise_pruned': 0,
        'sentiment_captured': 0,
        'final_records': 0,
        'accepted': 0,
    }
    drop_examples = {
        'noise': Counter(),
        'abstract': Counter(),
        'pattern': Counter(),
        'generic': Counter(),
        'sentiment': Counter(),
        'docfreq_high': Counter(),
        'docfreq_low': Counter(),
        'wordnet': Counter(),
        'praise': Counter(),
    }
    keybert_model = get_keybert(embedder)
    candidate_top_n = max(settings.keybert_top_n * 5, settings.keybert_top_n + 5)
    records: List[Dict[str, Any]] = []
    phrase_counter: Counter = Counter()
    doc_freq: Counter = Counter()

    allowed_high_freq = ASPECT_NOUN_WHITELIST.union(HOTEL_DOMAIN_NOUNS)

    def register_drop(reason: str, phrase: str) -> None:
        stats_key = {
            'noise': 'noise_pruned',
            'abstract': 'abstract_pruned',
            'pattern': 'pattern_pruned',
            'generic': 'generic_pruned',
            'sentiment': 'sentiment_captured',
            'praise': 'praise_pruned',
            'wordnet': 'wordnet_pruned',
            'sense': 'sense_pruned'
        }.get(reason)
        if stats_key:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        if reason in drop_examples:
            drop_examples[reason].update([phrase])

    for row in df.itertuples():
        raw_text = getattr(row, 'raw_text', None)
        normalized_text = getattr(row, 'normalized_text', None)
        clean_text = getattr(row, 'clean_text', None)
        document_text = raw_text or normalized_text or clean_text
        if not document_text:
            continue
        try:
            candidates = keybert_model.extract_keywords(
                document_text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=candidate_top_n,
            )
        except Exception:
            candidates = []
        stats['raw_candidates'] += len(candidates)
        filtered_candidates: List[Dict[str, Any]] = []
        seen_phrases: Set[str] = set()
        for phrase_raw, score in candidates:
            normalized_phrase = re.sub(r'\s+', ' ', phrase_raw.lower().strip())
            if not normalized_phrase or normalized_phrase in seen_phrases:
                continue
            seen_phrases.add(normalized_phrase)
            tokens = [re.sub(r'[^a-z]+', '', token.lower()) for token in word_tokenize(normalized_phrase)]
            tokens = [token for token in tokens if token]
            if not tokens:
                stats['pattern_pruned'] += 1
                drop_examples['pattern'].update([normalized_phrase])
                continue
            token_count = len(tokens)
            if token_count == 1:
                stats['raw_unigrams'] += 1
            elif token_count == 2:
                stats['raw_bigrams'] += 1
            else:
                stats['raw_trigrams'] += 1
            if score < settings.keybert_min_cosine:
                stats['similarity_pruned'] += 1
                continue
            if any(token in PRAISE_TARGETS for token in tokens):
                register_drop('praise', normalized_phrase)
                continue
            tagged = pos_tag(tokens)
            if not any(tag.startswith('NN') for _, tag in tagged):
                register_drop('pattern', normalized_phrase)
                continue
            if any(token in NOISE_NOUNS for token in tokens):
                register_drop('noise', normalized_phrase)
                continue
            abstract_hit = False
            for token in tokens:
                if token in ABSTRACT_IDEA_NOUNS or is_abstract_lemma(token):
                    abstract_hit = True
                    break
            if abstract_hit:
                register_drop('abstract', normalized_phrase)
                continue
            if all(token in GENERIC_PHRASE_COMPONENTS for token in tokens):
                register_drop('generic', normalized_phrase)
                continue
            invalid_pattern = False
            sentiments: List[str] = []
            for idx, (token, tag) in enumerate(tagged):
                if tag.startswith('JJ'):
                    if token not in FUNCTIONAL_ADJECTIVES:
                        register_drop('sentiment', normalized_phrase)
                        invalid_pattern = True
                        break
                    sentiments.append(token)
                elif not tag.startswith('NN'):
                    invalid_pattern = True
                    break
            if invalid_pattern:
                continue
            allowed_noun = any(token in allowed_high_freq or is_concrete_noun(token) for token, tag in tagged if tag.startswith('NN'))
            if not allowed_noun:
                senses = max(noun_synset_count(token) for token, tag in tagged if tag.startswith('NN'))
                if senses < settings.min_single_word_senses:
                    register_drop('sense', normalized_phrase)
                    continue
            if all(not wordnet.synsets(token) for token in tokens):
                register_drop('wordnet', normalized_phrase)
                continue
            filtered_candidates.append({
                'phrase': normalized_phrase,
                'score': float(score),
                'sentiments': sentiments,
            })
        if not filtered_candidates:
            continue
        filtered_candidates.sort(key=lambda item: item['score'], reverse=True)
        selected = filtered_candidates[:settings.keybert_top_n]
        if not selected:
            continue
        doc_phrase_set = set()
        for candidate in selected:
            phrase = candidate['phrase']
            doc_phrase_set.add(phrase)
            phrase_counter[phrase] += 1
            records.append({
                'source_id': getattr(row, 'source_id'),
                'phrase': phrase,
                'score': candidate['score'],
                'sentiments': candidate['sentiments'],
            })
        for phrase in doc_phrase_set:
            doc_freq[phrase] += 1

    if not records:
        raise RuntimeError('No keyword records extracted')

    stats['unique_candidates'] = len(phrase_counter)
    stats['top_candidates'] = phrase_counter.most_common(settings.diagnostic_top_keywords)

    final_records: List[Dict[str, Any]] = []
    for record in records:
        phrase = record['phrase']
        tokens = phrase.split()
        if len(tokens) > 1 and phrase_counter[phrase] < settings.min_ngram_freq:
            stats['low_freq_pruned'] += 1
            drop_examples['docfreq_low'].update([phrase])
            continue
        ratio = doc_freq[phrase] / stats['documents']
        if ratio > settings.max_phrase_doc_freq and not any(token in allowed_high_freq for token in tokens):
            stats['high_docfreq_pruned'] += 1
            drop_examples['docfreq_high'].update([phrase])
            continue
        final_records.append(record)

    stats['post_freq_candidates'] = len({record['phrase'] for record in final_records})
    stats['final_records'] = len(final_records)
    stats['accepted'] = len(final_records)
    stats['sentiment_pairs'] = 0
    stats['drop_examples'] = {reason: counter.most_common(10) for reason, counter in drop_examples.items()}
    for reason, counter in stats['drop_examples'].items():
        if counter:
            logger.info('Top %s drops: %s', reason, counter)

    return pd.DataFrame.from_records(final_records), stats

def deduplicate_keywords(df, settings, embedder):
    unique_phrases = df["phrase"].unique().tolist()
    vectors = encode_phrases(unique_phrases, embedder)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    sims, idxs = index.search(normalized, k=5)
    canonical = {}
    for i, phrase in enumerate(unique_phrases):
        canonical.setdefault(phrase, phrase)
        for sim, neighbor_idx in zip(sims[i], idxs[i]):
            if neighbor_idx == i:
                continue
            if sim >= settings.dedup_similarity_threshold:
                canonical[phrase] = canonical.get(unique_phrases[neighbor_idx], unique_phrases[neighbor_idx])
                break

    df = df.copy()
    df["canonical_phrase"] = df["phrase"].map(canonical)
    records = []
    for (source_id, canonical_phrase), group in df.groupby(["source_id", "canonical_phrase"], as_index=False):
        record = {'source_id':source_id,  'phrase':canonical_phrase, 
         'score':float(group["score"].max())}
        if "sentiments" in group.columns:
            sentiment_set = set()
            for entry in group["sentiments"]:
                if isinstance(entry, (list, tuple, set)):
                    sentiment_set.update(entry)

            record["sentiments"] = sorted(sentiment_set)
        records.append(record)

    deduped = pd.DataFrame(records)
    logger.info("Deduplicated to %s unique phrases", deduped["phrase"].nunique())
    return deduped


def reduce_vectors(vectors: "np.ndarray", target_dim: "int"=50) -> "np.ndarray":
    if vectors.size == 0:
        return vectors
    if vectors.shape[0] <= 1 or vectors.shape[1] <= target_dim:
        return vectors
    n_components = min(target_dim, vectors.shape[1], vectors.shape[0] - 1)
    if n_components <= 0:
        return vectors
    try:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(vectors)
        return reduced.astype(np.float32)
    except Exception:
        return vectors


def cluster_keywords(df: "pd.DataFrame", settings: "Settings") -> "Tuple[np.ndarray, float, str, float, Dict[int, np.ndarray]]":
    if df.empty:
        return (np.array([], dtype=int), 0.0, 'none', float('inf'), {})
    vectors = np.vstack(df['embedding'].to_list()).astype(np.float32)
    if vectors.shape[0] <= 1:
        return (np.zeros(vectors.shape[0], dtype=int), 0.0, 'singleton', float('inf'), {})
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vectors = vectors / norms
    reduced = reduce_vectors(vectors)

    def _evaluate_clusters(data: np.ndarray, labels: np.ndarray) -> "Tuple[float, float]":
        valid_mask = labels >= 0
        unique_labels = {int(lbl) for lbl in labels[valid_mask]}
        if len(unique_labels) <= 1 or valid_mask.sum() < 2:
            return (0.0, float('inf'))
        try:
            sil = float(silhouette_score(data[valid_mask], labels[valid_mask]))
        except Exception:
            sil = 0.0
        try:
            dbi = float(davies_bouldin_score(data[valid_mask], labels[valid_mask]))
        except Exception:
            dbi = float('inf')
        return (sil, dbi)

    def _run_kmeans(data: np.ndarray) -> "Tuple[np.ndarray, float, float]":
        max_k = min(settings.cluster_max_k, max(settings.cluster_min_k, int(math.sqrt(len(data)) * 2)))
        best_labels = None
        best_score = -1.0
        best_dbi = float('inf')
        for k in range(settings.cluster_min_k, max_k + 1):
            if k >= len(data):
                break
            labels = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=settings.random_seed).fit_predict(data)
            if len(set(labels)) <= 1:
                continue
            score, dbi = _evaluate_clusters(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels.astype(int)
                best_dbi = dbi
        if best_labels is None:
            best_labels = np.zeros(len(data), dtype=int)
            best_score = 0.0
            best_dbi = float('inf')
        return best_labels, float(best_score), float(best_dbi)

    def _run_hdbscan(data: np.ndarray) -> "Tuple[np.ndarray, float, float]":
        if not HAS_HDBSCAN or hdbscan is None:
            raise RuntimeError('HDBSCAN is not available')
        min_samples = settings.hdbscan_min_samples or settings.hdbscan_min_cluster_size
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(data).astype(int)
        score, dbi = _evaluate_clusters(data, labels)
        return labels, float(score), float(dbi)

    best_labels, best_score, best_dbi = _run_kmeans(reduced)
    method_used = 'kmeans'

    def _should_switch(score: float, dbi: float) -> bool:
        return score < settings.silhouette_switch_threshold or dbi > settings.davies_switch_threshold

    switch_to_hdbscan = False
    if settings.cluster_algorithm == 'hdbscan':
        switch_to_hdbscan = True
    elif settings.cluster_algorithm == 'auto':
        switch_to_hdbscan = _should_switch(best_score, best_dbi)

    if switch_to_hdbscan and HAS_HDBSCAN:
        try:
            labels_hdb, score_hdb, dbi_hdb = _run_hdbscan(reduced)
            positive_clusters = {int(lbl) for lbl in labels_hdb if int(lbl) >= 0}
            if positive_clusters and len(positive_clusters) > 2 and score_hdb >= settings.silhouette_fallback:
                best_labels, best_score, best_dbi = labels_hdb, score_hdb, dbi_hdb
                method_used = 'hdbscan'
            else:
                logger.debug(
                    'HDBSCAN results not adopted (clusters=%s, silhouette=%.3f, dbi=%.3f); keeping KMeans',
                    len(positive_clusters),
                    score_hdb,
                    dbi_hdb,
                )
        except RuntimeError:
            logger.warning('HDBSCAN requested but not available; falling back to KMeans')
    elif switch_to_hdbscan and not HAS_HDBSCAN:
        logger.warning('HDBSCAN requested but not available; falling back to KMeans')

    final_labels = best_labels.astype(int)
    counter = Counter(final_labels)
    valid_clusters = {cid for cid, count in counter.items() if cid != -1 and count >= settings.cluster_min_size}
    if not valid_clusters and counter:
        largest = max(counter.items(), key=lambda item: item[1])[0]
        if largest != -1:
            valid_clusters.add(largest)

    if valid_clusters:
        centroid_vectors = {
            cid: vectors[np.where(final_labels == cid)[0]].mean(axis=0)
            for cid in set(final_labels)
            if cid != -1
        }
        centroid_norms = {cid: vec / (np.linalg.norm(vec) or 1.0) for cid, vec in centroid_vectors.items()}
        for idx, label in enumerate(final_labels):
            if label in valid_clusters:
                continue
            if not centroid_norms:
                final_labels[idx] = -1
                continue
            best_cid = None
            best_sim = -1.0
            for cid, centroid_vec in centroid_norms.items():
                sim = float(np.dot(normalized_vectors[idx], centroid_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid
            if best_cid is not None:
                final_labels[idx] = best_cid

    counter = Counter(final_labels)
    for cid, count in list(counter.items()):
        if cid == -1:
            continue
        if count < settings.cluster_min_size:
            final_labels[final_labels == cid] = -1
    counter = Counter(final_labels)
    if len(counter) == 1 and -1 in counter:
        final_labels = np.zeros_like(final_labels)

    final_silhouette, final_dbi = _evaluate_clusters(reduced, final_labels)
    centroids: Dict[int, np.ndarray] = {}
    for cid in set(final_labels):
        if cid < 0:
            continue
        indices = np.where(final_labels == cid)[0]
        if indices.size == 0:
            continue
        centroids[int(cid)] = vectors[indices].mean(axis=0).astype(np.float32)

    logger.info('Selected clustering method %s with silhouette %.3f and DBI %.3f', method_used, final_silhouette, final_dbi)
    return final_labels.astype(int), float(final_silhouette), method_used, float(final_dbi), centroids



def clean_cluster_noise(cluster_df, labels, centroids, rules, settings):
    if labels.size == 0:
        return (
         labels, centroids, {}, False)
    else:
        vectors = np.vstack(cluster_df["embedding"].to_list()).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        forced_single_cluster = False
        if labels.size:
            if np.all(labels == -1):
                labels = np.zeros_like(labels, dtype=int)
                centroids = {}
                forced_single_cluster = True
        cluster_info = {}
        valid_clusters = set()
        rule_threshold = settings.rule_filter_min_confidence
        generic_threshold = 0.85
        for cid in set(labels):
            idxs = np.where(labels == cid)[0]
            size = int(len(idxs))
            if size == 0:
                continue
            info = {"original_size": size}
            if cid == -1:
                info.update({'original_rule_pass_ratio':0.0,  'original_noise_ratio':1.0,  'original_generic_ratio':1.0})
                cluster_info[int(cid)] = info
                continue
            phrases = cluster_df.iloc[idxs]["phrase"].tolist()
            confidences = []
            pass_count = 0
            generic_count = 0
            for phrase in phrases:
                if is_generic_phrase(phrase):
                    generic_count += 1
                match = apply_domain_rules(phrase, rules)
                conf = float(match["confidence"]) if match else 0.0
                confidences.append(conf)
                if conf >= rule_threshold:
                    pass_count += 1

            pass_ratio = pass_count / size if size else 0.0
            noise_ratio = 1.0 - pass_ratio
            generic_ratio = generic_count / size if size else 1.0
            info.update({'original_rule_pass_ratio':pass_ratio, 
             'original_noise_ratio':noise_ratio, 
             'original_generic_ratio':generic_ratio, 
             'max_confidence':max(confidences) if confidences else 0.0, 
             'min_confidence':min(confidences) if confidences else 0.0})
            cluster_info[int(cid)] = info
            if generic_ratio < generic_threshold and pass_ratio >= 0.0:
                valid_clusters.add(int(cid))

        if not valid_clusters:
            return (labels.astype(int), centroids, cluster_info, forced_single_cluster)
    merges_applied = False
    for cid, info in list(cluster_info.items()):
        cid_int = int(cid)
        if cid_int < 0:
            continue
        idxs = np.where(labels == cid_int)[0]
        size = idxs.size
        if size == 0:
            continue
        generic_ratio = info.get("original_generic_ratio", 1.0)
        noise_ratio = info.get("original_noise_ratio", 1.0)
        best_target = None
        best_sim = -1.0
        for candidate in valid_clusters:
            if candidate == cid_int:
                continue
            centroid_vec = centroids.get(candidate)
            if centroid_vec is None:
                continue
            centroid_norm = centroid_vec / (np.linalg.norm(centroid_vec) or 1.0)
            sims = normalized[idxs] @ centroid_norm
            mean_sim = float(np.mean(sims)) if sims.size else -1.0
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_target = candidate

        needs_merge = False
        merge_reason = None
        if generic_ratio >= 1.0:
            needs_merge = True
            merge_reason = "generic"
        else:
            if noise_ratio >= settings.cluster_noise_threshold:
                needs_merge = True
                merge_reason = "noisy"
            else:
                if size < settings.cluster_min_size:
                    needs_merge = True
                    merge_reason = "small"
                if needs_merge:
                    if best_target is not None and best_sim >= 0.45:
                        labels[idxs] = best_target
                        info["merged_into"] = int(best_target)
                        info["merge_reason"] = merge_reason
                    else:
                        labels[idxs] = -1
                        info["merged_into"] = -1
                        info["merge_reason"] = merge_reason
                    merges_applied = True
                    continue
        if best_target is not None and size < settings.cluster_min_size * 2 and best_sim >= 0.55:
            labels[idxs] = best_target
            info["merged_into"] = int(best_target)
            merges_applied = True
        else:
            info["merged_into"] = cid_int

    new_centroids = {}
    final_info = {}
    for cid in set(labels):
        idxs = np.where(labels == cid)[0]
        size = int(len(idxs))
        if cid >= 0:
            if idxs.size:
                new_centroids[int(cid)] = vectors[idxs].mean(axis=0).astype(np.float32)
        if size == 0:
            continue
        phrases = cluster_df.iloc[idxs]["phrase"].tolist()
        pass_count = 0
        generic_count = 0
        for phrase in phrases:
            if is_generic_phrase(phrase):
                generic_count += 1
            match = apply_domain_rules(phrase, rules)
            if match and float(match["confidence"]) >= rule_threshold:
                pass_count += 1

        pass_ratio = pass_count / size if size else 0.0
        generic_ratio = generic_count / size if size else 1.0
        final_info[int(cid)] = {'final_size':size, 
         'final_rule_pass_ratio':pass_ratio, 
         'final_noise_ratio':1.0 - pass_ratio, 
         'final_generic_ratio':generic_ratio}

    for cid, info in final_info.items():
        original = cluster_info.get(cid, {})
        original.update(info)
        cluster_info[int(cid)] = original

    merges_applied = merges_applied or forced_single_cluster
    return (labels.astype(int), new_centroids, cluster_info, merges_applied)


def _noun_phrase_candidates(phrase: "str") -> "List[str]":
    tokens = word_tokenize(phrase)
    if not tokens:
        return []
    tagged = pos_tag(tokens)
    candidates = []
    current = []
    for token, tag in tagged + [('', '')]:
        if tag.startswith("NN") or tag.startswith("JJ"):
            current.append(token)

    return [candidate for candidate in candidates if candidate]


def _normalise_label_text(candidate: "str") -> "str":
    tokens = word_tokenize(candidate)
    if not tokens:
        return ""
    tagged = pos_tag(tokens)
    filtered = []
    seen = set()
    for token, tag in tagged:
        lowered = token.lower()
        if tag.startswith("NN"):
            if lowered not in seen:
                filtered.append(lowered)
                seen.add(lowered)
            elif lowered in FUNCTIONAL_ADJECTIVES:
                if tag.startswith("JJ") and lowered not in seen:
                    filtered.append(lowered)
                    seen.add(lowered)

    if not filtered:
        for token, tag in tagged:
            if tag.startswith("NN"):
                lowered = token.lower()
                if lowered not in seen:
                    filtered.append(lowered)
                    seen.add(lowered)

    return " ".join(filtered)


def label_clusters(df, labels, score_map=None, top_n=5):
    if len(labels) == 0:
        return {}
    temp = df.reset_index(drop=True).copy()
    temp["cluster"] = labels
    label_lookup = {}
    for cid, group in temp.groupby("cluster"):
        cid_int = int(cid)
        if cid_int == -1:
            label_lookup[cid_int] = "Noise"
            continue
        else:
            phrases = group["phrase"].tolist()
            if not phrases:
                label_lookup[cid_int] = f"Cluster {cid_int}"
                continue
            ranked_phrases = sorted(phrases,
              key=(lambda phrase: (
             score_map.get(phrase, 0.0) if score_map else 0.0, len(phrase))),
              reverse=True)
            top_phrases = ranked_phrases[:top_n]
            candidate_counter = Counter()
            for phrase in top_phrases:
                for candidate in _noun_phrase_candidates(phrase):
                    normalized = _normalise_label_text(candidate)
                    if normalized:
                        candidate_counter[normalized] += 1

            if candidate_counter:
                label_raw = max((candidate_counter.items()),
                  key=(lambda item: (
                 item[1], len(item[0]))))[0]
            else:
                label_raw = _normalise_label_text(top_phrases[0]) or top_phrases[0].lower()
        label_lookup[cid_int] = label_raw.title()

    return label_lookup


def combine_embeddings(embedding: "Iterable[float]", centroid: "Iterable[float]") -> "np.ndarray":
    emb = np.asarray(embedding, dtype=(np.float32))
    centroid_arr = np.asarray(centroid, dtype=(np.float32))
    if centroid_arr.size == 0:
        centroid_arr = emb
    if centroid_arr.shape != emb.shape:
        centroid_arr = np.resize(centroid_arr, emb.shape)
    return np.concatenate([emb, centroid_arr], axis=0)


def prepare_feature_matrix(df: "pd.DataFrame") -> "np.ndarray":
    features = []
    for row in df.itertuples():
        centroid = getattr(row, "cluster_centroid", row.embedding)
        features.append(combine_embeddings(row.embedding, centroid))

    if features:
        return np.vstack(features)
    return np.empty((0, 0), dtype=(np.float32))


def build_cluster_training_samples(domain_df: "pd.DataFrame", cluster_centroids: "Dict[int, np.ndarray]") -> "pd.DataFrame":
    if domain_df.empty or not cluster_centroids:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    records = []
    for cid, group in domain_df.groupby('cluster'):
        cid_int = int(cid)
        if cid_int < 0:
            continue
        centroid = cluster_centroids.get(cid_int)
        if centroid is None:
            continue
        try:
            majority_kpi = group['domain_kpi'].mode().iat[0]
        except Exception:
            majority_kpi = group['domain_kpi'].iloc[0]
        try:
            majority_department = group['domain_department'].mode().iat[0]
        except Exception:
            majority_department = group['domain_department'].iloc[0]
        records.append({
            'phrase': f"cluster_{cid_int}_centroid",
            'embedding': centroid.tolist(),
            'cluster_centroid': centroid.tolist(),
            'kpi': majority_kpi,
            'department': majority_department,
            'cluster': cid_int,
        })
    return pd.DataFrame(records)

def build_hard_negative_samples(domain_df: "pd.DataFrame", cluster_centroids: "Dict[int, np.ndarray]") -> "pd.DataFrame":
    if domain_df.empty or len(cluster_centroids) <= 1:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    records = []
    items = list(cluster_centroids.items())
    for row in domain_df.itertuples():
        if getattr(row, "cluster", -1) < 0:
            continue
        alternatives = [(cid, centroid) for cid, centroid in items if cid != row.cluster]
        if not alternatives:
            continue
        cid_alt, centroid_alt = random.choice(alternatives)
        records.append({
            'phrase': f"{row.phrase}__neg_{cid_alt}",
            'embedding': list(row.embedding),
            'cluster_centroid': centroid_alt.tolist(),
            'kpi': '__noise__',
            'department': 'Unknown',
            'cluster': cid_alt,
        })

    return pd.DataFrame(records)

def build_seed_training(embedder: "SentenceTransformer", rules: "List[Dict[str, Any]]") -> "pd.DataFrame":
    seed_phrases = []
    for rule in rules:
        candidates = [
         rule["primary_phrase"]]
        candidates.extend(rule["keywords"][:3])
        for phrase in candidates:
            cleaned = phrase.strip()
            if not cleaned:
                continue
            seed_phrases.append({'phrase':cleaned, 
             'kpi':rule["kpi"], 
             'department':rule["department"]})

    if not seed_phrases:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    phrases = [entry["phrase"] for entry in seed_phrases]
    vectors = encode_phrases(phrases, embedder)
    for entry, vector in zip(seed_phrases, vectors):
        vec = vector.astype(np.float32)
        entry["embedding"] = vec.tolist()
        entry["cluster_centroid"] = vec.tolist()
        entry["cluster"] = -1

    df = pd.DataFrame(seed_phrases)
    logger.info("Prepared %s seed samples for KPI classifier", len(df))
    return df


def build_rule_training(df: "pd.DataFrame", settings: "Settings") -> "pd.DataFrame":
    if df.empty or "domain_confidence" not in df.columns:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    subset = df[df["domain_confidence"] >= settings.rule_training_confidence]
    if subset.empty:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    if "embedding" not in subset.columns:
        return pd.DataFrame(columns=['phrase', 'embedding', 'cluster_centroid', 'kpi', 'department', 'cluster'])
    records = []
    for row in subset.itertuples():
        embedding = list(row.embedding)
        centroid = list(getattr(row, "cluster_centroid", row.embedding))
        records.append({'phrase':row.phrase, 
         'embedding':embedding, 
         'cluster_centroid':centroid, 
         'kpi':row.domain_kpi, 
         'department':row.domain_department, 
         'cluster':getattr(row, "cluster", -1)})

    logger.info("Augmented training with %s high-confidence rule matches", len(records))
    return pd.DataFrame(records)


def train_kpi_classifier(training: "pd.DataFrame", settings: "Settings") -> "Tuple[LGBMClassifier, Dict[str, Any]]":
    if training.empty:
        raise RuntimeError('Training data for KPI classifier is empty')
    training = training.drop_duplicates(subset=['phrase', 'kpi', 'cluster'])
    mapping = {label: idx for idx, label in enumerate(sorted(training['kpi'].unique()))}
    inv_mapping = {idx: label for label, idx in mapping.items()}
    X = prepare_feature_matrix(training)
    if X.size == 0:
        raise RuntimeError('No feature vectors available for KPI classifier')
    y = training['kpi'].map(mapping).to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    have_multiple_classes = len(unique) > 1
    can_stratify = have_multiple_classes and counts.min() >= 2
    if len(training) > 4 and have_multiple_classes:
        test_size = 0.25 if len(training) >= 40 else 0.2
        stratify_vector = y if can_stratify else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=settings.random_seed,
            stratify=stratify_vector,
        )
    else:
        X_train, X_valid, y_train, y_valid = X, X, y, y

    class_values, class_counts = np.unique(y_train, return_counts=True)
    class_weights = {cls: len(y_train) / (len(class_values) * count) for cls, count in zip(class_values, class_counts)}
    sample_weights = np.array([class_weights[val] for val in y_train], dtype=np.float32)

    param_grid = [
        {'n_estimators': 120, 'learning_rate': 0.1, 'num_leaves': 63, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20},
        {'n_estimators': 160, 'learning_rate': 0.05, 'num_leaves': 96, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_samples': 30},
        {'n_estimators': 90, 'learning_rate': 0.15, 'num_leaves': 48, 'max_depth': 6, 'subsample': 0.75, 'colsample_bytree': 0.75, 'min_child_samples': 15},
    ]

    if HAS_OPTUNA:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 32, 128),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            }
            model = LGBMClassifier(objective='multiclass', num_class=len(mapping), n_jobs=-1, **params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = model.predict(X_valid)
            return f1_score(y_valid, preds, average='macro')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        best_params = study.best_params
    else:
        best_score = -1.0
        best_params = param_grid[0]
        for candidate in param_grid:
            model = LGBMClassifier(objective='multiclass', num_class=len(mapping), n_jobs=-1, **candidate)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            preds = model.predict(X_valid)
            score = f1_score(y_valid, preds, average='macro')
            if score > best_score:
                best_score = score
                best_params = candidate
        logger.info('Optuna unavailable; using LightGBM params %s', best_params)

    model = LGBMClassifier(objective='multiclass', num_class=len(mapping), n_jobs=-1, **best_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    preds = model.predict(X_valid)
    precision, recall, f1, _ = precision_recall_fscore_support(y_valid, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(y_valid, preds)
    valid_labels = sorted(set(np.unique(y_valid)) | set(np.unique(preds)))
    cm = confusion_matrix(y_valid, preds, labels=valid_labels).tolist()
    label_names = [inv_mapping[idx] for idx in valid_labels]
    class_support = {inv_mapping[idx]: int((y_train == idx).sum()) for idx in np.unique(y_train)}
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'mapping': mapping,
        'inv_mapping': inv_mapping,
        'confusion_matrix': {'labels': label_names, 'matrix': cm},
        'class_support': class_support,
    }
    return model, metrics

def assign_kpis(df, model, metrics, rules, settings, kpi_default_departments):
    if df.empty:
        return pd.DataFrame()
    feature_matrix = prepare_feature_matrix(df)
    if feature_matrix.size == 0:
        return pd.DataFrame()
    proba_model = model.predict_proba(feature_matrix)
    assignments = []
    mapping = metrics["mapping"]
    inv_mapping = metrics["inv_mapping"]
    noise_index = mapping.get("__noise__")
    for idx, row in df.iterrows():
        rule_match = apply_domain_rules(row["phrase"], rules)
        dist = proba_model[idx]
        adjusted = dist.copy()
        noise_prob = 0.0
        if noise_index is not None:
            if noise_index < len(adjusted):
                noise_prob = float(adjusted[noise_index])
                adjusted[noise_index] = -1.0
        best_idx = int(np.argmax(adjusted))
        model_kpi = inv_mapping.get(best_idx, next(iter(inv_mapping.values())))
        model_prob = float(dist[best_idx])
        if noise_index is not None:
            model_prob *= max(0.0, 1.0 - noise_prob)
        model_department = kpi_default_departments.get(model_kpi) or row.get("domain_department") or infer_department(model_kpi, re.findall("[a-zA-Z]+", model_kpi.lower()), list(kpi_default_departments.values()))
        rule_applied = False
        probability = model_prob
        kpi = model_kpi
        department = model_department
        if rule_match:
            rule_prob = rule_match['confidence']
            probability = max(rule_prob, model_prob)
            rule_stronger = rule_prob >= settings.rule_override_confidence and rule_prob >= model_prob
            model_stronger = model_prob >= settings.kpi_threshold and model_prob >= rule_prob
            if rule_stronger and rule_prob >= settings.rule_filter_min_confidence:
                kpi = rule_match['kpi']
                department = rule_match['department']
                rule_applied = True
            elif model_stronger:
                kpi = model_kpi
                department = model_department
                rule_applied = False
            else:
                kpi = rule_match['kpi']
                department = rule_match['department']
                rule_applied = True
        if kpi == "__noise__":
            probability = min(probability, 0.0)
            rule_applied = False
        sentiments = row.get("sentiments", [])
        accepted = probability >= settings.kpi_threshold and kpi != "__noise__" and noise_prob < 0.5
        assignments.append({'source_id':row["source_id"], 
         'phrase':row["phrase"], 
         'score':(row.get)("score", 0.0), 
         'embedding':row["embedding"], 
         'cluster_centroid':(row.get)("cluster_centroid", row["embedding"]), 
         'cluster':(row.get)("cluster", -1), 
         'kpi':kpi, 
         'department':department, 
         'probability':float(probability), 
         'accepted':accepted, 
         'rule_applied':rule_applied, 
         'noise_probability':noise_prob, 
         'sentiments':sentiments if (isinstance(sentiments, list)) else []})

    return pd.DataFrame(assignments)


def summarise_clusters(assignments: "pd.DataFrame", cluster_labels: "Dict[int, str]") -> "List[Dict[str, Any]]":
    clusters = []
    if assignments.empty:
        return clusters
    for cid, group in assignments.groupby("cluster"):
        cid_int = int(cid)
        label = cluster_labels.get(cid_int, f"Cluster {cid_int}")
        phrases = group["phrase"].tolist()
        non_generic = sum((1 for phrase in phrases if not is_generic_phrase(phrase)))
        generic_ratio = 1.0 - non_generic / max(len(phrases), 1)
        keywords = []
        for row in group.sort_values("probability", ascending=False).itertuples():
            keywords.append({'phrase':row.phrase, 
             'kpi':row.kpi, 
             'department':row.department, 
             'probability':float(row.probability), 
             'rule_applied':bool(row.rule_applied), 
             'sentiments':row.sentiments if (hasattr(row, "sentiments")) else []})

        clusters.append({'cluster_id':cid_int, 
         'label':label, 
         'keywords':keywords, 
         'generic_ratio':float(generic_ratio)})

    return clusters


def determine_output_filename(output_dir: "Path", settings: "Settings") -> "Path":
    if settings.output_file:
        custom = Path(settings.output_file)
        if not custom.is_absolute():
            custom = output_dir / custom
        return custom
    indices = []
    for path in output_dir.glob("*_keywords.json"):
        match = re.match("(\\d+)_keywords\\.json$", path.name)
        if match:
            indices.append(int(match.group(1)))

    base = settings.output_serial_start
    next_index = max(indices + [base - 1]) + 1 if indices else base
    while next_index in indices:
        next_index += 1

    return output_dir / f"{next_index}_keywords.json"


def export_output(summary, assignments, cluster_labels, settings):
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = determine_output_filename(output_dir, settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'run_summary':summary, 
     'clusters':summarise_clusters(assignments, cluster_labels)}
    output_path.write_text(json.dumps(payload, indent=2))
    return str(output_path)


def print_clusters(assignments, cluster_labels, silhouette_score_val):
    print("\n=== Keyword Clusters (silhouette %.3f) ===" % silhouette_score_val)
    if assignments.empty:
        print("No keyword assignments above threshold.")
        return
    for cluster in summarise_clusters(assignments, cluster_labels):
        if not cluster["keywords"]:
            continue
        print(f'\nCluster {cluster["cluster_id"]} - {cluster["label"]}')
        for kw in cluster["keywords"][:15]:
            flag = "RULE" if kw["rule_applied"] else "MODEL"
            sentiments = kw.get("sentiments") or []
            sentiment_txt = f" | sentiment={','.join(sentiments[:3])}" if sentiments else ""
            print(f'  - {kw["phrase"]} | KPI={kw["kpi"]} | dept={kw["department"]} | prob={kw["probability"]:.2f}{sentiment_txt} [{flag}]')


def log_gpu_status() -> "None":
    try:
        import torch
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            logger.info("GPU detected: %s", device_name)
        else:
            logger.info("No GPU detected; running on CPU")
    except ImportError:
        logger.info("PyTorch not available; running on CPU")


def run_pipeline(settings: "Settings") -> "Dict[str, Any]":
    log_gpu_status()
    embedder = get_embedder(settings)
    kpi_names, department_names = load_domain_metadata(settings)
    domain_rules = generate_domain_rules(kpi_names, department_names, settings)
    kpi_department_lookup = {rule['kpi']: rule['department'] for rule in domain_rules}

    diagnostics: Dict[str, Any] = {}

    reviews = ingest_reviews(settings)
    keywords_df, keyword_stats = extract_keywords(reviews, settings, embedder)
    diagnostics['keyword_extraction'] = keyword_stats
    logger.info('Top keyword candidates: %s', keyword_stats.get('top_candidates', [])[:5])

    deduped = deduplicate_keywords(keywords_df, settings, embedder)
    if deduped.empty:
        return {'run_name': settings.run_name, 'output_path': None, 'total_keywords': 0}

    unique_phrases = deduped['phrase'].unique().tolist()
    phrase_vectors = encode_phrases(unique_phrases, embedder)
    phrase_lookup = {phrase: vector.astype(np.float32) for phrase, vector in zip(unique_phrases, phrase_vectors)}
    phrase_score_map = deduped.groupby('phrase')['score'].max().to_dict()

    deduped = deduped.copy()
    deduped['embedding'] = deduped['phrase'].map(lambda phrase: phrase_lookup[phrase].tolist())

    cluster_input = pd.DataFrame({
        'phrase': unique_phrases,
        'embedding': [phrase_lookup[phrase].tolist() for phrase in unique_phrases],
        'score': [phrase_score_map.get(phrase, 0.0) for phrase in unique_phrases],
    })

    cluster_assignments, silhouette_val, cluster_method, cluster_dbi, cluster_centroids = cluster_keywords(cluster_input, settings)
    cluster_assignments, cluster_centroids, cluster_rule_info, merges_applied = clean_cluster_noise(
        cluster_input,
        cluster_assignments,
        cluster_centroids,
        domain_rules,
        settings,
    )
    if merges_applied:
        cluster_method = f"{cluster_method}+denoise"
    cluster_input['cluster'] = cluster_assignments
    cluster_sizes = Counter(cluster_assignments)
    noise_ratio = cluster_sizes.get(-1, 0) / max(len(cluster_assignments), 1)

    cluster_label_map = label_clusters(
        cluster_input,
        cluster_assignments,
        score_map=phrase_score_map,
        top_n=settings.diagnostic_clusters_top_n,
    )
    phrase_cluster_map = {phrase: int(label) for phrase, label in zip(unique_phrases, cluster_assignments)}
    deduped['cluster'] = deduped['phrase'].map(lambda phrase: phrase_cluster_map.get(phrase, -1))

    def _centroid_for_phrase(row: "pd.Series") -> "List[float]":
        centroid = cluster_centroids.get(int(row['cluster']))
        base_vec = phrase_lookup[row['phrase']]
        if centroid is None:
            centroid = base_vec
        return centroid.astype(np.float32).tolist()

    deduped['cluster_centroid'] = deduped.apply(_centroid_for_phrase, axis=1)

    domain_filtered = annotate_domain_matches(deduped, domain_rules, settings.rule_filter_min_confidence)
    domain_filtered = domain_filtered[domain_filtered['cluster'] >= 0].reset_index(drop=True)
    domain_stats = {
        'matched_rows': int(len(domain_filtered)),
        'matched_unique_phrases': int(domain_filtered['phrase'].nunique()),
        'filtered_out': int(len(deduped) - len(domain_filtered)),
    }
    diagnostics['domain_filter'] = domain_stats
    logger.info('Domain filter retained %d/%d phrase-doc pairs', domain_stats['matched_rows'], len(deduped))
    if domain_filtered.empty:
        return {'run_name': settings.run_name, 'output_path': None, 'total_keywords': 0}

    domain_filtered['embedding'] = domain_filtered['embedding'].apply(lambda vec: vec if isinstance(vec, list) else list(vec))
    domain_filtered['cluster_centroid'] = domain_filtered['cluster_centroid'].apply(lambda vec: vec if isinstance(vec, list) else list(vec))

    cluster_training = build_cluster_training_samples(domain_filtered, cluster_centroids)
    hard_negatives = build_hard_negative_samples(domain_filtered, cluster_centroids)
    seed_training = build_seed_training(embedder, domain_rules)
    rule_training = build_rule_training(domain_filtered, settings)

    for name, df in [('seed', seed_training), ('rule', rule_training), ('cluster', cluster_training), ('neg', hard_negatives)]:
        logger.debug('Training frame %s type=%s', name, type(df))
    training_frames = [df for df in (seed_training, rule_training, cluster_training, hard_negatives) if not df.empty]
    training = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()

    model, metrics = train_kpi_classifier(training, settings)
    assignments = assign_kpis(domain_filtered, model, metrics, domain_rules, settings, kpi_department_lookup)
    accepted_assignments = assignments[assignments['accepted']].reset_index(drop=True)

    cluster_highlights = {
        int(cid): sorted(
            [phrase for phrase in cluster_input.loc[cluster_input['cluster'] == cid, 'phrase']],
            key=lambda phrase: phrase_score_map.get(phrase, 0.0),
            reverse=True,
        )[: settings.diagnostic_clusters_top_n]
        for cid in set(cluster_assignments)
        if int(cid) >= 0
    }

    logger.info('Cluster summary: method=%s silhouette=%.3f DBI=%.3f noise=%.1f%%', cluster_method, silhouette_val, cluster_dbi, noise_ratio * 100)
    top_cluster_sizes = list(itertools.islice(sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True), 10))
    logger.info('Cluster size distribution (top10): %s', top_cluster_sizes)
    noise_preview = {
        int(cid): round(info.get('final_noise_ratio', info.get('original_noise_ratio', 0.0)), 3)
        for cid, info in cluster_rule_info.items()
        if cid >= 0
    }
    top_noise = list(itertools.islice(sorted(noise_preview.items(), key=lambda item: item[1], reverse=True), 5))
    logger.info('Cluster rule-noise ratios (top5): %s', top_noise)
    logger.info('Training samples: seeds=%d rules=%d clusters=%d negatives=%d', len(seed_training), len(rule_training), len(cluster_training), len(hard_negatives))

    diagnostics['clustering'] = {
        'method': cluster_method,
        'silhouette': float(silhouette_val),
        'davies_bouldin': float(cluster_dbi),
        'noise_ratio': noise_ratio,
        'sizes': {int(cid): int(count) for cid, count in cluster_sizes.items()},
        'rule_noise': cluster_rule_info,
    }
    diagnostics['training'] = {
        'seed_rows': len(seed_training),
        'rule_rows': len(rule_training),
        'cluster_rows': len(cluster_training),
        'negative_rows': len(hard_negatives),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1': metrics.get('f1'),
        'accuracy': metrics.get('accuracy'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'class_support': metrics.get('class_support'),
    }

    summary = {
        'run_name': settings.run_name,
        'silhouette': float(silhouette_val),
        'cluster_davies_bouldin': float(cluster_dbi),
        'davies_bouldin': float(cluster_dbi),
        'classifier_precision': metrics.get('precision'),
        'classifier_recall': metrics.get('recall'),
        'classifier_f1': metrics.get('f1'),
        'classifier_accuracy': metrics.get('accuracy'),
        'cluster_method': cluster_method,
        'total_keywords': int(len(accepted_assignments)),
        'retained_candidates': int(len(domain_filtered)),
        'threshold': settings.kpi_threshold,
        'noise_ratio': noise_ratio,
        'cluster_count': int(sum(1 for cid in cluster_sizes if cid != -1)),
        'cluster_sizes': {str(int(cid)): int(count) for cid, count in cluster_sizes.items()},
        'cluster_highlights': cluster_highlights,
        'diagnostics': diagnostics,
    }

    output_path = export_output(summary, accepted_assignments, cluster_label_map, settings)
    summary['output_path'] = output_path

    print_clusters(accepted_assignments, cluster_label_map, silhouette_val)
    print("\\nResults saved to", output_path)


def parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(description="Domain-aware keyword extraction pipeline")
    parser.add_argument("--input_csv", type=str, default=(settings.input_csv), help="Path to reviews CSV")
    parser.add_argument("--text_column", type=str, default=(settings.text_column), help="Column with review text")
    parser.add_argument("--id_column", type=str, default=(settings.id_column), help="Column with document IDs")
    parser.add_argument("--kpi_metadata", type=str, default=(settings.kpi_metadata_path), help="CSV listing KPI names")
    parser.add_argument("--department_metadata", type=str, default=(settings.department_metadata_path), help="CSV listing department names")
    parser.add_argument("--output_dir", type=str, default=(settings.output_dir), help="Directory for JSON output")
    parser.add_argument("--output_file", type=str, default=None, help="Optional explicit output filename")
    parser.add_argument("--report_only", action="store_true", help="Print report without writing JSON")
    return parser.parse_args()


def main() -> "None":
    args = parse_args()
    settings.input_csv = args.input_csv
    settings.text_column = args.text_column
    settings.id_column = args.id_column
    settings.kpi_metadata_path = args.kpi_metadata
    settings.department_metadata_path = args.department_metadata
    settings.output_dir = args.output_dir
    if args.output_file:
        settings.output_file = args.output_file
    summary = run_pipeline(settings)
    if args.report_only:
        if summary.get("output_path"):
            try:
                Path(summary["output_path"]).unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
