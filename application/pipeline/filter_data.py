import numpy as np
import pandas as pd

def filter(dataset):
    documents = dataset['train']['document']
    document_lengths = [len(doc.split()) for doc in documents]
    Q1 = np.percentile(document_lengths, 25)
    Q3 = np.percentile(document_lengths, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    def is_within_bounds(example):
        doc_length = len(example['document'].split())
        return lower_bound <= doc_length <= upper_bound

    return dataset.filter(is_within_bounds)