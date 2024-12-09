import numpy as np
import spacy_universal_sentence_encoder
from datasets import load_dataset
from joblib._multiprocessing_helpers import mp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')


def get_embeddings(sentences):
    embeddings = np.array([nlp(sentence).vector for sentence in sentences])
    return embeddings


def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences


def cluster_sentences(sentences, num_clusters=10):
    embeddings = get_embeddings(sentences)  # Use batch processing
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans


def select_representative_sentences(sentences, clusters, kmeans):
    cluster_centers = kmeans.cluster_centers_
    representative_sentences = []

    embeddings = get_embeddings(sentences)

    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - cluster_centers[i], axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representative_sentences.append(sentences[closest_idx])

    return representative_sentences


def extract_sentences_with_k_means(documents):

    doc = documents.replace('|||||', " ")
    doc = doc.replace('\n \n', " ")

    sentences = preprocess_text(doc)

    num_clusters= 10
    num_clusters = min(num_clusters, len(sentences))

    clusters, kmeans = cluster_sentences(sentences, num_clusters=num_clusters)

    summary_sentences = select_representative_sentences(sentences, clusters, kmeans)

    return " ".join(summary_sentences)

def balanced_general_trunk_summaries_preprocess(example, token_limit=1024, word_token_ratio=1.5):
    start_prompt = "summarize:"
    end_prompt = ""

    summaries = example['extractive_summary'].replace("[DOC]", ".")
    prompt = start_prompt + summaries + end_prompt

    print("PROMPT:", prompt)

    return prompt


def create_inputs(batch):
    with mp.Pool(mp.cpu_count()) as pool:
        summarized_examples = pool.map(extract_sentences_with_k_means, batch['document'])

    prompts = [balanced_general_trunk_summaries_preprocess({'extractive_summary': summary}) for summary in
               summarized_examples]

    return {
        'input': prompts,
        'labels': batch['summary']
    }

def preprocess_data_with_k_means(dataset):
    return dataset.map(create_inputs, batched=True, num_proc=1)