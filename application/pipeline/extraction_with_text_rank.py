import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import re

from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained sentence embedding model
stop_words = set(stopwords.words('english'))


def read_article(text):
    sentences = sent_tokenize(text)

    cleaned_sentences = [re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", sentence)).strip() for sentence in sentences]

    return cleaned_sentences


def build_similarity_matrix(sentences):
    # Embed sentences using a pre-trained model (e.g., BERT/Sentence-BERT)
    sentence_embeddings = model.encode(sentences)

    # Calculate cosine similarity matrix for embedded sentences
    similarity_matrix = cosine_similarity(sentence_embeddings)

    # Optional: Introduce a similarity threshold to filter out weak connections
    threshold = 0.3  # You can tune this value
    similarity_matrix[similarity_matrix < threshold] = 0

    return similarity_matrix


def extract_sentences(text, top_n=10):
    sentences = read_article(text)

    if len(sentences) == 0:
        return "", 0

    sentence_similarity_matrix = build_similarity_matrix(sentences)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize_text = [ranked_sentences[i][1] for i in range(min(len(ranked_sentences), top_n))]

    return ".".join(summarize_text), len(sentences)


def extract_sentences_with_text_rank(documents):
    doc = documents.replace('|||||', ". ")
    sentences, _ = extract_sentences(doc)
    return sentences


def balanced_general_trunk_summaries_preprocess(example, token_limit=1024, word_token_ratio=1.5):
    start_prompt = "summarize:"
    end_prompt = ""

    summaries = example['extractive_summary'].replace("[DOC]", ".")
    prompt = start_prompt + summaries + end_prompt

    return prompt


def create_inputs(batch):
    with mp.Pool(mp.cpu_count()) as pool:
        summarized_examples = pool.map(extract_sentences_with_text_rank, batch['document'])

    prompts = [balanced_general_trunk_summaries_preprocess({'extractive_summary': summary}) for summary in
               summarized_examples]

    return {
        'input': prompts,
        'labels': batch['summary']
    }


def preprocess_data_with_text_rank(dataset):
    return dataset.map(create_inputs, batched=True, num_proc=1)
