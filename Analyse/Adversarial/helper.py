import re
import pandas as pd
import json
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def normalize(text):
    if text is None:
        return ""
    return " ".join(text.lower().strip().split())


def exact_match(pred, gold_list):
    pred_n = normalize(pred)
    return int(any(pred_n == normalize(g) for g in gold_list))


def f1_score(pred, gold):
    pred_tokens = word_tokenize(pred.lower())
    gold_tokens = word_tokenize(gold.lower())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def max_f1(pred, gold_list):
    return max(f1_score(pred, g) for g in gold_list)

def categorize(em, f1):
    if em == 1:
        return "correct"
    elif f1 >= 0.5:
        return "partially correct"
    else:
        return "failed"


def load_and_process_predictions(file_path, category_type):
    rows = []
    with open(file_path) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if category_type!= None:
        df["multi_question"] = df["question"].apply(category_type)
    
    df["exact_match"] = df.apply(lambda r: exact_match(r["predicted_answer"], r["answers"]["text"]), axis=1)
    df["f1"] = df.apply(lambda r: max_f1(r["predicted_answer"], r["answers"]["text"]), axis=1)
    df["result_type"] = df.apply(lambda r: categorize(r["exact_match"], r["f1"]), axis=1)

    if category_type!= None:
        summary = df.groupby(["multi_question", "result_type"]).size().unstack(fill_value=0)
    else:
        summary = df.groupby(["result_type"]).size()#.unstack(fill_value=0)
    # print(summary)
    return summary


