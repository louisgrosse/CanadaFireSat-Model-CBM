#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single-word, semantically meaningful vocabulary from a CSV/JSONL/Parquet
dictionary of captions/phrases. Streams the data and shows tqdm progress bars by default.

Now with:
  - Optional PROPN dropping (on by default) to remove names like cities/people.
  - Optional NER filtering (GPE/LOC/FAC by default) to remove places/stadiums/etc.
  - Expanded generic ban list for non-visual and sports/media terms (e.g., "provided", "series", "stadium").

Outputs:
  - msclip_words.csv  with columns: word, pos, count, example
  - msclip_words.txt  one word per line, sorted by frequency desc
"""

from __future__ import annotations
import argparse
import csv
import sys
import re
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from collections import Counter

import pandas as pd
from tqdm import tqdm

# ---- Optional dependency: spaCy -------------------------------------------------
try:
    import spacy
except Exception as e:
    spacy = None

DEFAULT_ALLOWED_POS = ["NOUN", "ADJ"]
DEFAULT_MIN_LEN = 3
DEFAULT_MIN_COUNT = 1

# Generic "boring" words and domain-irrelevant stuff (can be disabled via --no-nonvisual-ban)
BAN_NONVISUAL = {
    # function/helper-ish nouns/verbs/adjectives that don't help wildfire/scene semantics
    "provided","including","series","various","several","multiple","misc","generic","random","type","types","kind","kinds",
    "sort","sorts","range","ranges","size","sizes","amount","amounts","number","numbers","percent","percentage",
    "data","dataset","sample","samples","example","examples","collection","collections","copyright","license","source","credit",
    "official","homepage","website","link","page","twitter","instagram","facebook","reddit","wiki","http","https","www","com","org","net",
    # location/administrative generics (keep 'mountain', 'snow' etc.)
    "city","cities","town","towns","village","villages","district","districts","province","provinces","state","states","county","counties",
    "capital","capitals","municipality","municipalities","metropolis","metropolitan",
    # sports/media
    "stadium","stadiums","stadia","arena","arenas","court","courts","pitch","pitches","match","matches","tournament","tournaments",
    "league","leagues","cup","season","seasons","final","finals","episode","episodes","series","score","scores","team","teams","fc","united",
    "club","clubs","broadcast","broadcaster","news","media","channel","channels","tv","radio","show","shows",
}

# Extra tokens to always drop (case-insensitive)
ALWAYS_DROP = {
    "the","a","an","and","or","but","if","then","else","when","where","while","for","to","of","by",
    "on","in","at","from","with","without","into","onto","about","above","below","under","over",
    "between","among","before","after","during","until","within","through","throughout","via",
    "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "having","can","could","may","might","must","shall","should","will","would","as","than","too",
    "very","not","no","nor","so","such","this","that","these","those","here","there","it","its",
    "he","she","they","them","him","her","you","your","yours","we","us","our","ours","me","my","mine",
    "who","whom","whose","which","what","where","why","how"
}

# Media/imagery generics (can be noisy)
GENERIC_BAN = {
    "area","areas","region","regions","site","sites","zone","zones","view","views","scene","scenes","image","images","imagery","landscape","landscapes",
    "photo","photos","picture","pictures","closeup","close-ups","close-up",
    "satellite","aerial","shot","footage","frame","graphic","illustration","vector","render","clip","stock","background"
}


def _load_spacy(lang: str, need_ner: bool):
    if spacy is None:
        raise RuntimeError(
            "spaCy is not installed. Please `pip install spacy` and download a model, e.g. "
            "`python -m spacy download en_core_web_sm`."
        )
    disable = ["textcat", "senter"]
    if not need_ner:
        disable.append("ner")
    nlp = spacy.load(lang, disable=disable)
    nlp.max_length = max(nlp.max_length, 4_000_000)
    return nlp


def _iter_frames(path: Path, columns: Optional[List[str]], chunksize: int = 100_000) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for df in pd.read_csv(path, usecols=columns, chunksize=chunksize):
            yield df
    elif suffix in {".tsv", ".tab"}:
        for df in pd.read_csv(path, sep="\t", usecols=columns, chunksize=chunksize):
            yield df
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=columns)
        yield df
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=True)
        if columns:
            df = df[columns]
        yield df
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def _infer_text_columns(df: pd.DataFrame) -> List[str]:
    obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    return obj_cols or list(df.columns)


def _normalize_token(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z'\-]", "", t)
    t = re.sub(r"-{2,}", "-", t)
    return t


def generate_words(
    df: pd.DataFrame,
    nlp,
    columns: List[str],
    allowed_pos: List[str],
    drop_set: set,
    min_len: int,
    n_process: int,
    show_progress: bool,
    drop_propn: bool,
    banned_ner_types: set,
) -> Iterable[Tuple[str, str, str]]:
    """Yield (lemma, pos, example_phrase) for each kept token."""
    texts = (df[columns].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
    iterator = nlp.pipe(texts, n_process=n_process, batch_size=2048)
    if show_progress:
        iterator = tqdm(iterator, total=len(texts), desc="spaCy", dynamic_ncols=True, mininterval=0.5, leave=False)

    for doc in iterator:
        seen_in_this_doc = set()
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            if tok.is_stop:
                continue
            if tok.like_num:
                continue
            if getattr(tok, "is_quote", False) or tok.is_currency or tok.is_bracket:
                continue

            pos = tok.pos_
            if drop_propn and pos == "PROPN":
                continue
            if allowed_pos and pos not in allowed_pos:
                continue
            if banned_ner_types and tok.ent_type_ and tok.ent_type_ in banned_ner_types:
                continue

            lemma = tok.lemma_.lower() if tok.lemma_ else tok.text.lower()
            lemma = _normalize_token(lemma)
            if not lemma or len(lemma) < min_len:
                continue
            if lemma in drop_set:
                continue
            if any(ch.isdigit() for ch in lemma):
                continue
            if lemma in seen_in_this_doc:
                continue
            seen_in_this_doc.add(lemma)
            yield (lemma, pos, doc.text)


def collect_vocab(
    input_paths: List[Path],
    columns: Optional[List[str]],
    out_dir: Path,
    lang: str,
    allowed_pos: List[str],
    min_len: int,
    min_count: int,
    extra_stop: Optional[Path],
    n_workers: int,
    show_progress: bool,
    drop_propn: bool,
    banned_ner_types: set,
    use_nonvisual_ban: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    need_ner = bool(banned_ner_types)
    nlp = _load_spacy(lang, need_ner=need_ner)

    # Build stop set
    stop_set = set(ALWAYS_DROP)
    stop_set |= set(GENERIC_BAN)
    if use_nonvisual_ban:
        stop_set |= set(BAN_NONVISUAL)
    stop_set |= set(getattr(nlp.Defaults, "stop_words", set()))
    if extra_stop and extra_stop.exists():
        with extra_stop.open("r", encoding="utf-8") as f:
            stop_set |= {ln.strip().lower() for ln in f if ln.strip()}

    counts: Counter[str] = Counter()
    exemplar: Dict[str, str] = {}
    pos_store: Dict[str, str] = {}

    for p in input_paths:
        tqdm.write(f"Reading {p} (columns = {columns if columns else 'infer'})")
        # Infer columns if needed
        if columns is None:
            for df in _iter_frames(p, None, 20000):
                columns = _infer_text_columns(df)
                break
            tqdm.write(f"Inferred text columns: {columns}")

        for df in _iter_frames(p, columns, 20000):
            gen = generate_words(
                df=df,
                nlp=nlp,
                columns=columns,
                allowed_pos=allowed_pos,
                drop_set=stop_set,
                min_len=min_len,
                n_process=n_workers,
                show_progress=show_progress,
                drop_propn=drop_propn,
                banned_ner_types=banned_ner_types,
            )
            for lemma, pos, example in gen:
                counts[lemma] += 1
                if lemma not in exemplar:
                    exemplar[lemma] = example[:200]
                if lemma not in pos_store:
                    pos_store[lemma] = pos

    items = [(w, pos_store[w], c, exemplar[w]) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x[2], x[0]))

    out_csv = out_dir / "msclip_words_filtered.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "pos", "count", "example"])
        for w, p, c, e in items:
            writer.writerow([w, p, c, e])

    print(f"Wrote: {out_csv} (rows={len(items)})")
    return out_csv


def _parse_banned_ner_types(types_list: Optional[List[str]]) -> set:
    if not types_list:
        return {"GPE", "LOC", "FAC"}
    lowered = [t.strip().upper() for t in types_list]
    if len(lowered) == 1 and lowered[0] in {"NONE", "NO", "OFF"}:
        return set()
    return set(lowered)



def main():
    path = "/work/eceo/grosse/ssl4eo-s12/_caps_parquet/"
    input_paths = [Path(path+"train.parquet"), Path(path+"val.parquet")]

    out_csv = collect_vocab(
        input_paths=input_paths,
        columns=["caption", "question_1", "question_2", "question_3"],
        out_dir=Path("/work/eceo/grosse/dictionnary"),
        lang="en_core_web_sm",
        allowed_pos=DEFAULT_ALLOWED_POS,
        min_len=3,
        min_count=1,
        extra_stop=None,
        n_workers=8,
        show_progress=True,
        drop_propn=True,
        banned_ner_types=["GPE", "LOC", "FAC"],
        use_nonvisual_ban=True,
    )


if __name__ == "__main__":
    main()
