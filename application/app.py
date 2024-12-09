from flask import Flask, render_template, request, redirect
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from werkzeug.utils import secure_filename

from pipeline.extraction_with_k_means import extract_sentences_with_k_means
from pipeline.extraction_with_text_rank import balanced_general_trunk_summaries_preprocess, extract_sentences_with_text_rank
from pipeline.inference import generate_summary

# Initialize the app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_bart_name = 'sshleifer/distilbart-cnn-12-6'
model_t5_name = 'spacemanidol/flan-t5-small-3-6-cnndm'

tokenizer_bart = AutoTokenizer.from_pretrained(model_bart_name)
tokenizer_t5 = AutoTokenizer.from_pretrained(model_t5_name)
model_bart = AutoModelForSeq2SeqLM.from_pretrained(model_bart_name)
model_flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_t5_name)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def summarize_text(text, extractive_model, abstractive_model):
    if extractive_model == 'k-means':
        ext_summ = extract_sentences_with_k_means(text)
    else:
        ext_summ = extract_sentences_with_text_rank(text)
        print("EXT SUMM: ", ext_summ)

    if abstractive_model == 't5':
        model = model_flan_t5
        tokenizer = tokenizer_t5
    else:
        model = model_bart
        tokenizer = tokenizer_bart

    input = balanced_general_trunk_summaries_preprocess({'extractive_summary': ext_summ})
    print("INPUT: ", input)
    return generate_summary(input, model, tokenizer)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        extractive_model = request.form.get('algorithm')  # Get selected algorithm (k-means or TextRank)
        abstractive_model = request.form.get('model')  # Get selected model (Distill-BART or T5)

        files = request.files.getlist('file')
        file_contents = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()

                file_contents.append(text_content)

        text_to_summarize = "|||||".join(file_contents)

        summary = summarize_text(text_to_summarize, extractive_model, abstractive_model)

        return render_template('index.html', summary=summary)

    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
