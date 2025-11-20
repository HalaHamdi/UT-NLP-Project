import datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, \
    TrainingArguments, HfArgumentParser
from helpers import prepare_train_dataset_qa, QuestionAnsweringTrainer, adjust_output_dir
import argparse

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # Key TrainingArguments to specify when calling from command line:
    # --per_device_train_batch_size <int, default=8>: Training batch size
    # --num_train_epochs <float, default=3.0>: Number of passes through training data
    # --output_dir <path>: Where to save model checkpoints (required)
    # --learning_rate <float, default=5e-5>: Learning rate for training
    # --warmup_steps <int, default=0>: Number of warmup steps
    # --logging_steps <int, default=500>: Log every X updates steps
    # --save_steps <int, default=500>: Save checkpoint every X updates steps

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""Base model to fine-tune. Should be a HuggingFace model ID 
                      (see https://huggingface.co/models) or path to a saved checkpoint.""")
    argp.add_argument('--dataset', type=str, default='squad',
                      help="""Dataset to use. Default is 'squad'. Can also specify a custom 
                      JSON/JSONL file path.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""Maximum sequence length for training. 
                      Shorter lengths use less memory but may truncate examples.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of training examples.')

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
        train_split = 'train'
    else:
        # Load SQuAD or other dataset from HuggingFace
        dataset = datasets.load_dataset(args.dataset)
        train_split = 'train'
    
    # Initialize model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    
    # Make tensor contiguous if needed (for ELECTRA models)
    # https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare dataset preprocessing function
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)

    print("☑️Preprocessing training data... (this takes a little bit, should only happen once per dataset)")
    
    train_dataset = dataset[train_split]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names
    )

    # Initialize the QuestionAnsweringTrainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("✅ Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"✅ Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()