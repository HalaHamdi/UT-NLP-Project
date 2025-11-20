import datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, \
    TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_validation_dataset_qa, QuestionAnsweringTrainer, adjust_output_dir
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # Key TrainingArguments to specify when calling from command line:
    # --output_dir <path>: Where to save evaluation results (required)
    # --per_device_eval_batch_size <int, default=8>: Evaluation batch size

    argp.add_argument('--model', type=str, required=True,
                      help="""Path to the trained model checkpoint or HuggingFace model ID 
                      to evaluate.""")
    argp.add_argument('--dataset', type=str, default='squad',
                      help="""Dataset to evaluate on. Default is 'squad'. Can also specify 
                      a custom JSON/JSONL file path.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""Maximum sequence length for evaluation. 
                      Should match the length used during training.""")
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of evaluation examples.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Ensure outputs go under ./Data using shared helper in helpers.py
    original_output_dir = training_args.output_dir
    training_args.output_dir = adjust_output_dir(training_args.output_dir)
    if original_output_dir != training_args.output_dir:
        print(f"ℹ️ Adjusted output_dir from {original_output_dir} to {training_args.output_dir}")

    # Load dataset
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # The "json" loader places all examples in train split by default
        eval_split = 'train'
    else:
        # Load SQuAD or other dataset from HuggingFace
        dataset = datasets.load_dataset(args.dataset)
        eval_split = 'validation'
    
    # Load model and tokenizer from checkpoint
    print(f"✅ Loading model from {args.model}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare dataset preprocessing function
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    print("✅ Preprocessing evaluation data... (this takes a little bit, should only happen once per dataset)")
    
    eval_dataset = dataset[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    # Setup evaluation metrics
    metric = evaluate.load('squad')
    compute_metrics = lambda eval_preds: metric.compute(
        predictions=eval_preds.predictions, 
        references=eval_preds.label_ids
    )

    # Store predictions for later dumping
    eval_predictions = None
    
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the QuestionAnsweringTrainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    
    # Evaluate the model
    print(" ✅ Starting evaluation...")
    results = trainer.evaluate(eval_examples=eval_dataset)

    print('\n' + '='*50)
    print('✅ Evaluation Results:')
    print('='*50)
    for key, value in results.items():
        print(f'{key}: {value}')
    print('='*50 + '\n')

    # Save evaluation metrics
    os.makedirs(training_args.output_dir, exist_ok=True)

    metrics_path = os.path.join(training_args.output_dir, 'eval_metrics.json')
    with open(metrics_path, encoding='utf-8', mode='w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save predictions
    predictions_path = os.path.join(training_args.output_dir, 'eval_predictions.jsonl')
    with open(predictions_path, encoding='utf-8', mode='w') as f:
        predictions_by_id = {
            pred['id']: pred['prediction_text'] 
            for pred in eval_predictions.predictions
        }
        for example in eval_dataset:
            example_with_prediction = dict(example)
            example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
            f.write(json.dumps(example_with_prediction))
            f.write('\n')
    print(f"✅ Predictions saved to {predictions_path}")
    
    print("\n☑️ Evaluation complete!")


if __name__ == "__main__":
    main()