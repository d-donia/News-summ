import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PegasusForCausalLM, PegasusTokenizer


def load_model_tokenizer(model_name):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

