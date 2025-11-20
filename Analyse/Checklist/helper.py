from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from checklist.pred_wrapper import PredictorWrapper
import torch
import itertools
import json
import pandas as pd
from tabulate import tabulate
from IPython.display import display, Markdown
# answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx]))



def crossproduct(t):
    # takes the output of editor.template and does the cross product of contexts and qas
    ret = []
    ret_labels = []
    for x in t.data:
        cs = x['contexts']
        qas = x['qas']
        d = list(itertools.product(cs, qas))
        ret.append([(x[0], x[1][0]) for x in d])
        ret_labels.append([x[1][1] for x in d])
    t.data = ret
    t.labels = ret_labels
    return t

def format_squad_with_context(x, pred, conf, label=None, *args, **kwargs):
    c, q = x
    ret = 'C: %s\nQ: %s\n' % (c, q)
    if label is not None:
        ret += 'A: %s\n' % label
    ret += 'P: %s\n' % pred
    return ret

def get_finetuned_electra_predictor(model_path="../../Data/trained_model"):
    """
    Returns a wrapped predictor compatible with CheckList for your finetuned ELECTRA QA model.

    Input: list of (context, question) tuples
    Output: list of predicted answer strings (exact original text)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()  # set to evaluation mode

    def predict_fn(examples):
        """
        examples: list of (context, question) tuples
        returns: list of answer strings
        """
        answers = []
        for context, question in examples:
            # Encode inputs and get offsets for exact character mapping
            inputs = tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                return_offsets_mapping=True
            )
            offset_mapping = inputs.pop("offset_mapping")[0]  # shape: [seq_len, 2]

            with torch.no_grad():
                outputs = model(**inputs)

            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)

            # Use offsets to extract exact substring from context
            start_char = offset_mapping[start_idx][0].item()
            end_char = offset_mapping[end_idx][1].item()
            answer = context[start_char:end_char]

            if answer.strip() == "":
                answer = "empty"

            answers.append(answer)

        return answers

    # Wrap predictor for CheckList
    wrapped_predictor = PredictorWrapper.wrap_predict(predict_fn)
    return wrapped_predictor

def show_example(test_cases, n=1):
    model = get_finetuned_electra_predictor()
    for i in range(n):
        pred = model(test_cases.data[i])[0]
        labels = test_cases.labels[i]
        for j, item in enumerate(test_cases.data[i]):
            print(f"  [{j+1}] {item} , pred: {pred[j]} , label: {labels[j]}")
        print('---')


def export_suite_to_jsonl(suite, output_file="checklist_testsuite.jsonl"):
    all_rows = []

    for testname in suite.tests:                  
        test = suite.tests[testname]
        all_predictions = test.results.preds
        all_labels = test.labels

        for testcase,pred_array,label_array in zip(test.data,all_predictions, all_labels):

            for (context,question), pred, label in zip(testcase, pred_array, label_array):
            
                all_rows.append({
                    'test_name': testname,
                    'context': context,
                    'question': question,
                    'prediction': pred,
                    'label': label
                })

        

    with open(output_file, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Saved", len(all_rows), "rows to {output_file}.")
   
def get_summary(suite):
    summary_list = []

    for testname, test in suite.tests.items():
        stats = test.get_stats()
        n_examples = sum(len(group) for group in test.data)
        
        summary_list.append({
            "Test Name": testname,
            "Total Cases": stats.testcases,
            "Example Per Case": f"{n_examples/stats.testcases:.2f}",
            "Failures": stats.fails,
            "Failure Rate": f"{stats.fail_rate:.2f}%",
        })

    return pd.DataFrame(summary_list)


def display_and_export_mdtable(dataframe, do_display= True, do_export=True, output_file="checklist_testsuite_summary.md"):
    md_table = tabulate(dataframe, headers='keys', tablefmt='github', showindex=False)
    if do_display:
        display(Markdown(md_table))

    if do_export:
        with open(output_file, "w") as f:
            f.write(md_table)