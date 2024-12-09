from datasets import load_dataset, load_from_disk
from filter_data import filter
#from preprocess_colab import preprocess_data
from extraction_with_text_rank import preprocess_data_with_text_rank
from extraction_with_k_means import preprocess_data_with_k_means
from inference import generate_summary
from model_setup import load_model_tokenizer
from compute_metrics import compute_rouge, compute_bert
import numpy as np
from datasets import concatenate_datasets

# Function to calculate mean ROUGE and BERTScores
def evaluate_model(model_name, dataset):

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    bert_scores = []

    rouge_scores_s = {"rouge1": [], "rouge2": [], "rougeL": []}
    bert_scores_s = []


    model, tokenizer = load_model_tokenizer(model_name)

    for example in dataset:

        source = "summarize:" + example['document']
        sample = example['input']
        reference = example['summary']

        # Generate prediction
        output = generate_summary(sample, model=model, tokenizer=tokenizer)
        print("SAMPLE:", sample)
        print("-----------------------OUTPUT-K-----------------------" )
        print(output)
        print("------------------------------------------------------" )


        output_s = generate_summary(source, model=model, tokenizer=tokenizer)

        print("-----------------------OUTPUT-S-----------------------")
        print(output_s)
        print("------------------------------------------------------")

        rouge_result = compute_rouge(output, reference)
        rouge_scores['rouge1'].append(rouge_result["rouge1"])
        rouge_scores['rouge2'].append(rouge_result["rouge2"])
        rouge_scores['rougeL'].append(rouge_result["rougeL"])

        rouge_result = compute_rouge(output_s, reference)
        rouge_scores_s['rouge1'].append(rouge_result["rouge1"])
        rouge_scores_s['rouge2'].append(rouge_result["rouge2"])
        rouge_scores_s['rougeL'].append(rouge_result["rougeL"])

        # Compute BERTScore
        P, R, F1 = compute_bert(output, reference)
        bert_scores.append(F1)

        P, R, F1 = compute_bert(output_s, reference)
        bert_scores_s.append(F1)



    mean_rouge = {key: np.mean(values) for key, values in rouge_scores.items()}
    mean_bert = np.mean(bert_scores)

    mean_rouge_s = {key: np.mean(values) for key, values in rouge_scores_s.items()}
    mean_bert_s = np.mean(bert_scores_s)

    return mean_rouge, mean_bert, mean_rouge_s, mean_bert_s

def evaluate_dataset(extractive_module):
    model_names = ['sshleifer/distilbart-cnn-12-6', 'spacemanidol/flan-t5-small-3-6-cnndm']

    dataset = load_dataset("multi_news")

    filtered_dataset = filter(dataset)


    if extractive_module == 'k-means':
        processed_dataset = preprocess_data_with_k_means(filtered_dataset)
    else:
        processed_dataset = preprocess_data_with_text_rank(filtered_dataset)


    train_dataset = processed_dataset['train']
    validation_dataset = processed_dataset['validation']
    test_dataset = processed_dataset['test']

    merged_dataset = concatenate_datasets([train_dataset, validation_dataset, test_dataset])

    print(merged_dataset)


    results = {}
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        mean_rouge, mean_bert, mean_rouge_s, mean_bert_s = evaluate_model(model_name, merged_dataset)
        results[model_name] = {
            "mean_rouge": mean_rouge,
            "mean_bert_score": mean_bert,
            "mean_rouge_s": mean_rouge_s,
            "mean_bert_score_s": mean_bert_s,
        }

    print(results)
