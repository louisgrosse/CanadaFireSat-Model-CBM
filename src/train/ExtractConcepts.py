import pandas as pd
import re
from pathlib import Path
from collections import Counter
import spacy

ROOT = Path("/work/eceo/grosse/ssl4eo-s12/_caps_parquet")
OUT_DIR = Path("/work/eceo/grosse/dictionnary") 
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 100000        
MIN_COUNT = 5            
TOP_N = 50000            
print("Loading parquet...")


df_train = pd.read_parquet(ROOT / "train.parquet")
df_val   = pd.read_parquet(ROOT / "val.parquet")
df = pd.concat([df_train, df_val], ignore_index=True)

text_cols = [c for c in ["caption","question_1","question_2","question_3"] if c in df.columns]
df = df[text_cols].fillna("").astype(str)

rows = (df.apply(lambda r: " ".join(r.values.tolist()), axis=1)).tolist()
print(f"Total rows available: {len(rows)}")


nlp = spacy.load("en_core_web_sm", disable=["ner"]) 

bad_tail_words = set(["area","region","site","zone","view","scene","image","imagery","landscape"])
ban_phrases_exact = {
    "satellite",
    "satellite image",
    "satellite imagery",
    "aerial",
    "aerial imagery",
    "aerial view",
}

def normalize_phrase(tokens):
    lemmas = [t.lemma_.lower().strip() for t in tokens if not t.is_punct]

    while lemmas and lemmas[0] in ("the","a","an","this","that"):
        lemmas = lemmas[1:]
    while lemmas and lemmas[-1] in bad_tail_words:
        lemmas = lemmas[:-1]

    phrase = " ".join(lemmas)
    phrase = re.sub(r"\s+", " ", phrase).strip()

    if len(phrase) < 3:
        return None
    if phrase in ban_phrases_exact:
        return None
    return phrase

def extract_phrases(doc):
    phrases = []

    for chunk in doc.noun_chunks:
        norm = normalize_phrase(list(chunk))
        if norm:
            phrases.append(norm)

    # (B) adjective/noun spans: contiguous ADJ/NOUN/PROPN ending in a NOUN/PROPN
    for sent in doc.sents:
        tags = [(t.lemma_, t.pos_) for t in sent]
        start = None
        for i, (_, pos) in enumerate(tags):
            if pos in ("ADJ","NOUN","PROPN"):
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i
                    if tags[end-1][1] in ("NOUN","PROPN"):
                        span_tokens = sent[start:end]
                        norm = normalize_phrase(list(span_tokens))
                        if norm:
                            phrases.append(norm)
                start = None
        if start is not None:
            end = len(tags)
            if tags[end-1][1] in ("NOUN","PROPN"):
                span_tokens = sent[start:end]
                norm = normalize_phrase(list(span_tokens))
                if norm:
                    phrases.append(norm)

    return phrases

print("Extracting phrases...")
freq = Counter()
for idx, text in enumerate(rows):
    if idx >= MAX_ROWS:
        break
    if not text.strip():
        continue
    doc = nlp(text)
    phs = extract_phrases(doc)
    freq.update(phs)

print(f"Processed {idx+1} rows.")
print(f"Unique raw phrases: {len(freq)}")

def canonicalize(p):
    p = p.strip()
    p = re.sub(r"\b(forest|woodland|grassland|shrubland) (area|region|zone)$", r"\1", p)
    p = re.sub(r"neighbourhood","neighborhood",p)
    p = re.sub(r"neighbours","neighbors",p)
    return p

merged = Counter()
for phrase, c in freq.items():
    merged[canonicalize(phrase)] += c

ban_single = set(["area","land","field","forest","water","road","building","vegetation"])

final_items = []
for phrase, c in merged.items():
    if c < MIN_COUNT:
        continue
    if phrase in ban_single:
        continue
    if " " not in phrase and len(phrase) <= 5:
        # e.g. "road", "farm", "tree": too generic
        continue
    final_items.append((phrase, c))

# sort
final_items.sort(key=lambda x: (-x[1], x[0]))

top_items = final_items[:TOP_N]

# save outputs
out_counts = OUT_DIR / "concept_counts50k_spacy.csv"

pd.DataFrame(final_items, columns=["concept","count"]).to_csv(out_counts, index=False)

print("Total usable concepts:", len(final_items))
print("Top 10 concepts:", top_items[:10])
print("Saved:")
print(" ", out_counts)
