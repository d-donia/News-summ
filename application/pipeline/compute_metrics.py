import evaluate

from bert_score import BERTScorer

rouge = evaluate.load('rouge')

def compute_rouge(predicted_summary, reference_summary):
    rouge_result = rouge.compute(predictions=[predicted_summary], references=[reference_summary])

    return rouge_result

def compute_bert(predicted_summary, reference_summary):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([predicted_summary], [reference_summary])

    return P,R,F1

