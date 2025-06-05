import argparse
import csv
import json
import os
import random
import sys
import time

import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --------------- CONFIGURATION ---------------

# Adjust this if your Flask app is served elsewhere
API_URL = "http://localhost:5001/chat"

# Path to the JSON file containing questions and expected answers
TESTSET_PATH = os.path.join(os.path.dirname(__file__), "testset.json")

# Directory where CSV results will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
DETAILED_CSV = os.path.join(OUTPUT_DIR, "detailed_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_stats.csv")


# --------------- HELPER FUNCTIONS ---------------

def load_testset(path):
    """
    Reads testset.json and returns a list of dicts:
      [ {"id": ..., "question": ..., "answer": ...}, ... ]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions_and_answers", [])


def sample_questions(all_items, num=None):
    """
    If num is None or >= total items, return all items.
    Otherwise, return `num` items chosen uniformly at random.
    """
    if num is None or num >= len(all_items):
        return all_items.copy()
    return random.sample(all_items, k=num)


def compute_exact_match(reference: str, hypothesis: str) -> int:
    """
    Returns 1 if reference and hypothesis match exactly (case-insensitive), else 0.
    """
    return 1 if reference.strip().lower() == hypothesis.strip().lower() else 0


def compute_word_overlap(reference: str, hypothesis: str) -> float:
    """
    Computes the ratio of overlapping unigrams between reference and hypothesis:
      overlap = # shared unigrams / # unigrams in reference
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0.0
    shared = set(ref_tokens).intersection(hyp_tokens)
    return len(shared) / len(set(ref_tokens))


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Computes BLEU-4 score (with smoothing) for a single reference-hypothesis pair.
    Using NLTK's sentence_bleu with SmoothingFunction.
    """
    ref_tokens = [reference.lower().split()]
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
    return score


def compute_rouge_1_f1(reference: str, hypothesis: str) -> float:
    """
    Computes ROUGE-1 F1 score for a single reference-hypothesis pair:
      P = (# overlapping unigrams) / (# unigrams in hypothesis)
      R = (# overlapping unigrams) / (# unigrams in reference)
      F1 = 2 * P * R / (P + R), or 0 if both numerators are zero.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts = {}
    for w in ref_tokens:
        ref_counts[w] = ref_counts.get(w, 0) + 1

    overlap_count = 0
    for w in set(hyp_tokens):
        if w in ref_counts:
            overlap_count += 1

    precision = overlap_count / len(hyp_tokens)
    recall = overlap_count / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def call_chat_api(question: str, lang: str = "en", retries=3, backoff=2.0):
    """
    Sends a POST request to API_URL with JSON {"message": question, "lang": lang}.
    Returns the 'reply' string. Retries up to `retries` times on failures, 
    with exponential backoff. 
    """
    payload = {"message": question, "lang": lang}
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            return body.get("reply", "")
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"  -> Warning: API call failed "
                      f"(attempt {attempt+1}/{retries}): {e!r}. Retrying in {wait:.1f}s…")
                time.sleep(wait)
            else:
                raise e


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_detailed_csv(records, path):
    """
    Writes a CSV where each row has:
      id, question, expected_answer, model_answer, exact_match, word_overlap,
      bleu_score, rouge1_f1
    """
    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "expected_answer",
                "model_answer",
                "exact_match",
                "word_overlap_ratio",
                "bleu_score",
                "rouge1_f1"
            ],
            delimiter=","
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r["id"],
                "question": r["question"],
                "expected_answer": r["expected_answer"],
                "model_answer": r["model_answer"],
                "exact_match": r["exact_match"],
                "word_overlap_ratio": f"{r['word_overlap_ratio']:.3f}".replace(".", ","),
                "bleu_score":           f"{r['bleu_score']:.3f}".replace(".", ","),
                "rouge1_f1":            f"{r['rouge1_f1']:.3f}".replace(".", ","),
            })


def write_summary_csv(records, path):
    """
    Writes a summary CSV with:
      total_questions, avg_exact_match, avg_word_overlap, avg_bleu, avg_rouge1_f1
    """
    total = len(records)
    if total == 0:
        avg_exact = 0.0
        avg_overlap = 0.0
        avg_bleu = 0.0
        avg_rouge = 0.0
    else:
        avg_exact = sum(r["exact_match"] for r in records) / total
        avg_overlap = sum(r["word_overlap_ratio"] for r in records) / total
        avg_bleu = sum(r["bleu_score"] for r in records) / total
        avg_rouge = sum(r["rouge1_f1"] for r in records) / total

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_questions", total])
        writer.writerow(["avg_exact_match", f"{avg_exact:.3f}"])
        writer.writerow(["avg_word_overlap_ratio", f"{avg_overlap:.3f}"])
        writer.writerow(["avg_bleu_score", f"{avg_bleu:.3f}"])
        writer.writerow(["avg_rouge1_f1", f"{avg_rouge:.3f}"])


# --------------- SCRIPT ENTRYPOINT ---------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: sample N questions from testset.json, "
                    "query the chat API in English, and save results with BLEU/ROUGE scores."
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=None,
        help="Number of random questions to run. If omitted or >= total, runs all."
    )
    args = parser.parse_args()

    # 1) Load all items from testset.json
    test_items = load_testset(TESTSET_PATH)
    if not test_items:
        print(f"No items found in {TESTSET_PATH}. Exiting.")
        sys.exit(1)

    # 2) Sample randomly (no fixed seed)
    chosen = sample_questions(test_items, args.num)
    print(f"Running evaluation on {len(chosen)} question(s).")

    # 3) Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)

    detailed_records = []
    for idx, item in enumerate(chosen, start=1):
        qid = item.get("id", "")
        question = item.get("question", "").strip()
        expected = item.get("answer", "").strip()

        print(f"[{idx}/{len(chosen)}] ID={qid}: \"{question}\" ... ", end="", flush=True)
        try:
            model_reply = call_chat_api(question, lang="en", retries=3, backoff=2.0)
        except Exception as e:
            print(f"ERROR calling API: {e!r}")
            model_reply = ""

        print("done.")

        # 4) Compute metrics
        em = compute_exact_match(expected, model_reply)
        wo = compute_word_overlap(expected, model_reply)
        bleu = compute_bleu(expected, model_reply)
        rouge1 = compute_rouge_1_f1(expected, model_reply)

        detailed_records.append({
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "model_answer": model_reply.strip(),
            "exact_match": em,
            "word_overlap_ratio": wo,
            "bleu_score": bleu,
            "rouge1_f1": rouge1
        })

    # 5) Write detailed CSV
    write_detailed_csv(detailed_records, DETAILED_CSV)
    print(f"Detailed results written to {DETAILED_CSV}")

    # 6) Write summary CSV
    write_summary_csv(detailed_records, SUMMARY_CSV)
    print(f"Summary stats written to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()