import argparse
import csv
import json
import os
import random
import sys
import time

import requests
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# CONFIG
API_URL          = "http://localhost:5001/chat"
TESTSET_PATH     = os.path.join(os.path.dirname(__file__), "testset.json")
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "evaluation_results")

EVALUATION_CSV   = os.path.join(OUTPUT_DIR, "evaluation_csv.csv")
SUMMARY_CSV      = os.path.join(OUTPUT_DIR, "summary_stats.csv")

# HELPERS 
def load_testset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("questions_and_answers", [])

def sample_questions(all_items, num=None):
    if num is None or num >= len(all_items):
        return all_items.copy()
    return random.sample(all_items, k=num)

def compute_exact_match(ref: str, hyp: str) -> int:
    return int(ref.strip().lower() == hyp.strip().lower())

def compute_word_overlap(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    return len(set(r) & set(h)) / len(set(r)) if r else 0.0

def compute_bleu(ref: str, hyp: str) -> float:
    refs, hyp_tokens = [ref.lower().split()], hyp.lower().split()
    if not hyp_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return sentence_bleu(refs, hyp_tokens, smoothing_function=smoothie)

def compute_rouge1_f1(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r or not h:
        return 0.0
    counts = {}
    for w in r:
        counts[w] = counts.get(w, 0) + 1
    overlap = sum(1 for w in set(h) if w in counts)
    p = overlap / len(h)
    recall = overlap / len(r)
    return 2 * p * recall / (p + recall) if (p + recall) else 0.0

def call_chat_api(question: str, lang="en", retries=3, backoff=2.0) -> str:
    payload = {"message": question, "lang": lang}
    headers = {"Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            res = requests.post(API_URL, json=payload, headers=headers)
            res.raise_for_status()
            return res.json().get("reply", "").strip()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(
                    f"  -> Warning: API call failed "
                    f"(attempt {attempt+1}/{retries}): {e!r}. Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
            else:
                raise

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Eval: sample N questions, call chat API, pivot on refs, export CSV."
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=None,
        help="Number of random questions to run (default=all)."
    )
    args = parser.parse_args()

    items = load_testset(TESTSET_PATH)
    if not items:
        print(f"No items in {TESTSET_PATH}, exiting.")
        sys.exit(1)

    chosen = sample_questions(items, args.num)
    print(f"Running evaluation on {len(chosen)} question(s).")
    ensure_output_dir(OUTPUT_DIR)

    rows = []
    for i, item in enumerate(chosen, start=1):
        qid      = item.get("id", "")
        question = item.get("question", "").strip()
        answers  = item.get("answers") or [item.get("answer") or ""]
        print(f"[{i}/{len(chosen)}] ID={qid}: collecting…", end="\r")

        try:
            model_ans = call_chat_api(question, lang="en")
        except Exception as e:
            print(f"\nERROR calling API: {e!r}")
            model_ans = ""

        for idx_ref, ref in enumerate(answers, start=1):
            rows.append({
                "id":           qid,
                "question":     question,
                "model_answer": model_ans,
                "ref_index":    idx_ref,
                "reference":    ref,
                "exact_match":  compute_exact_match(ref, model_ans),
                "word_overlap": round(compute_word_overlap(ref, model_ans) * 100, 1),
                "bleu":         round(compute_bleu(ref, model_ans) * 100, 1),
                "rouge1_f1":    round(compute_rouge1_f1(ref, model_ans) * 100, 1),
            })

    df = pd.DataFrame(rows)
    print(" " * 80, end="\r")  # overschrijf statusregel

    df = df.sort_values(["id", "bleu"], ascending=[True, False])

    pivot = df.pivot_table(
        index=["id", "question", "model_answer"],
        columns="ref_index",
        values=["reference", "exact_match", "word_overlap", "bleu", "rouge1_f1"],
        aggfunc="first"
    )
    pivot.columns = [f"{metric}_{idx}" for metric, idx in pivot.columns]
    pivot = pivot.reset_index()

    def sorted_cols(prefix):
        return sorted(
            [c for c in pivot.columns if c.startswith(prefix)],
            key=lambda x: int(x.rsplit("_", 1)[1])
        )

    ref_cols  = sorted_cols("reference_")
    bleu_cols = sorted_cols("bleu_")
    wo_cols   = sorted_cols("word_overlap_")
    em_cols   = sorted_cols("exact_match_")
    rf_cols   = sorted_cols("rouge1_f1_")

    base      = ["id", "question", "model_answer"]
    new_order = base + ref_cols + bleu_cols + wo_cols + em_cols + rf_cols
    pivot     = pivot[new_order]

    pivot.to_csv(EVALUATION_CSV, index=False, encoding="utf-8-sig")
    print(f"- Evaluation CSV: {EVALUATION_CSV}")

    total = len(df)
    if total:
        summary = {
            "total_pairs":      total,
            "avg_exact_match":  df["exact_match"].mean(),
            "avg_word_overlap": df["word_overlap"].mean(),
            "avg_bleu":         df["bleu"].mean(),
            "avg_rouge1_f1":    df["rouge1_f1"].mean(),
        }
    else:
        summary = dict.fromkeys(
            ["total_pairs", "avg_exact_match", "avg_word_overlap", "avg_bleu", "avg_rouge1_f1"],
            0.0
        )

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.2f}"])
    print(f"- Summary CSV: {SUMMARY_CSV}")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()