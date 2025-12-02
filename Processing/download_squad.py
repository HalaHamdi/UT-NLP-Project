from datasets import load_dataset
load_dataset("rajpurkar/squad", "plain_text", split="validation").to_json("squad_validation.jsonl", lines=True)