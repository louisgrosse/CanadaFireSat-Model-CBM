import sys
import os
import glob
from typing import List, Optional
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy

# ----------------------------------------------------------------------
# MS-CLIP setup
# ----------------------------------------------------------------------

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model_, preprocess, tokenizer = build_model(
    model_name="Llama3-MS-CLIP-Base", 
    pretrained=True,
    ckpt_path=None,
    device=device,
    channels=10,                       
)
model = model_.to(device).eval()

# ----------------------------------------------------------------------
# Label utilities: normalization + simple filters
# ----------------------------------------------------------------------

GENERIC_LABELS = {
    "area", "areas", "region", "regions", "zone", "zones",
    "image", "images", "scene", "scenes", "view", "views",
    "satellite image", "satellite view", "satellite imagery",
    "landscape", "landscape scene",
}

COLOR_WORDS = {
    "brown", "blue", "green", "black", "white", "grey", "gray",
    "red", "yellow", "orange",
}


def normalize_label_str(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def is_bad_label(s: str) -> bool:
    s = normalize_label_str(s)
    if not s:
        return True
    if s in GENERIC_LABELS:
        return True
    if s in COLOR_WORDS:
        return True
    if len(s) <= 2:
        return True
    return False


# ----------------------------------------------------------------------
# Load sentences (parquet captions) from the SSL4EO dataset
# ----------------------------------------------------------------------

def load_sentences_from_parquet(
    captions_dir: str,
    pattern: str = "*.parquet",
    max_phrases: Optional[int] = None,
) -> List[str]:
    paths = sorted(glob.glob(os.path.join(captions_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No parquet files found under {captions_dir} / {pattern}")

    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    phrases: List[str] = []
    taken = False
    for col in ["caption", "captions", "text", "description", "prompt"]:
        if col in df.columns:
            phrases.extend(df[col].dropna().astype(str).tolist())
            taken = True

    if not taken:
        for col in df.columns:
            if df[col].dtype == object:
                phrases.extend(df[col].dropna().astype(str).tolist())

    seen = set()
    uniq: List[str] = []
    for s in phrases:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if max_phrases is not None and len(uniq) > max_phrases:
        uniq = uniq[:max_phrases]

    print(f"[INFO] Loaded {len(uniq)} unique sentences.")
    return uniq


# ----------------------------------------------------------------------
# MS-CLIP sentence vs candidate phrases (KeyBert like)
# ----------------------------------------------------------------------

@torch.no_grad()
def highlight_phrases_with_msclip(
    sentence: str,
    candidates: List[str],
    top_k: int = 1,
    ngram: int = 5,
) -> List[str]:
    """
    Given one sentence and a list of candidate phrases:
      - encode the sentence with MS-CLIP
      - encode all candidates
      - return the top_k candidates with highest cosine similarity.
    """
    if not candidates:
        return []

    # encode sentence
    toks_sent = tokenizer([sentence]).to(device)
    sent_emb = model.inference_text(toks_sent)
    sent_emb = F.normalize(sent_emb, dim=-1)[0]  # [D]

    # encode candidates
    toks = tokenizer(candidates).to(device)
    cand_embs = model.inference_text(toks)
    cand_embs = F.normalize(cand_embs, dim=-1)   # [C, D]

    sims = cand_embs @ sent_emb                  # [C]

    counts = torch.tensor([len(s.split()) for s in candidates], device=sims.device)

    out = []
    for n in range(1, ngram + 1):
        idx = (counts == n).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue   # <-- skip, don't append []

        k_n = min(top_k, idx.numel())
        top_local = torch.topk(sims[idx], k=k_n).indices
        top_global = idx[top_local].cpu().tolist()
        out.extend([candidates[j] for j in top_global])


    return out


# ----------------------------------------------------------------------
# Term vocab via spaCy NP candidates
# ----------------------------------------------------------------------

def build_term_vocab_spacy(
    sentences: List[str],
    min_freq: int = 5,
    max_terms: Optional[int] = 100_000,
    max_ngram: int = 3,
    use_msclip: bool = False,
    top_k_per_sentence: int = 1,
    n_process: int = 1,
) -> List[str]:
    """
    SpaCy-based term extraction.

    If use_msclip=False (Idea 1):
      - For each sentence, extract 1–3-gram spans of (NOUN/PROPN/ADJ)* ending in NOUN/PROPN.
      - Filter with is_bad_label().
      - Count all surviving spans globally.

    If use_msclip=True (Idea 4):
      - Same candidates.
      - Use MS-CLIP to pick the top_k_per_sentence phrases most similar
        to the full sentence embedding, and only count those.
    """

    print("processees : ", n_process)
    print("[INFO] Loading spaCy model...")

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    counts: Counter = Counter()
    print(f"[INFO] Extracting terms with spaCy (use_msclip={use_msclip})...")
    # iterate sentences + docs together
    for sent, doc in tqdm(
        zip(sentences, nlp.pipe(sentences, batch_size=512, n_process=n_process)),
        total=len(sentences),
    ):

        # 1) candidate spans: noun-headed 1..max_ngram-grams
        content_toks = [
            tok for tok in doc
            if tok.is_alpha
            and not tok.is_stop
            and tok.pos_ in {"NOUN", "ADJ"}
        ]

        candidate_phrases: List[str] = []

        for n in range(1, max_ngram + 1):
            if n > max_ngram or len(content_toks) < n:
                continue
            for i in range(len(content_toks) - n + 1):
                span = content_toks[i:i+n]
                head = span[-1]
                
                lemmas = [t.lemma_.lower() for t in span]
                if len(set(lemmas)) != len(lemmas):
                    continue

                # require head to be noun/proper noun
                if head.pos_ not in {"NOUN"}:
                    continue

                phrase = normalize_label_str(" ".join(t.lemma_ for t in span))
                if is_bad_label(phrase):
                    continue

                # require at least one noun/proper noun inside
                if not any(t.pos_ in {"NOUN"} for t in span):
                    continue
                
                candidate_phrases.append(phrase)

        if not candidate_phrases:
            continue

        # 2) Count candidates
        if use_msclip:
            best_phrases = highlight_phrases_with_msclip(
                sent,
                candidate_phrases,
                top_k=top_k_per_sentence,
                ngram = max_ngram
            )
            for p in best_phrases:
                counts[p] += 1
        else:
            for p in candidate_phrases:
                counts[p] += 1

    # 3) Frequency filtering + sorting
    items = [(p, c) for p, c in counts.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])  # highest freq first

    if max_terms is not None and len(items) > max_terms:
        print("CAREFULLLLLLLL Cropped list")
        items = items[:max_terms]

    terms = [p for p, c in items]
    print(f"[INFO] Built spaCy term vocabulary of size {len(terms)} (min_freq={min_freq}).")
    return terms


def main():
    captions_dir = "/home/louis/Code/wildfire-forecast/dictionnary/"  # <- adjust
    out_dir = "/home/louis/Code/CanadaFireSat-Model-CBM/results/DictionnaryWordsGeneral"
    sentences = load_sentences_from_parquet(captions_dir)
    USE_MSCLIP_SALIENCY = True

    # --------- Build term vocabulary BEFORE k-means ----------
    terms = build_term_vocab_spacy(
        sentences,
        min_freq=1,
        max_terms=1_500_000,
        max_ngram=5,    
        use_msclip=USE_MSCLIP_SALIENCY,
        top_k_per_sentence=2,
        n_process=os.cpu_count(),
    )
    out_stub = "spacy" if not USE_MSCLIP_SALIENCY else "spacy_msclip"


    # --------- Save dictionary ----------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_stub}_general_f1k2n5_homogeneous.csv")

    df = pd.DataFrame({
        "concept": list(terms)
    })

    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved {len(terms)} concepts to {out_path}")

if __name__ == "__main__":
    main()
