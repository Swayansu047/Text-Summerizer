import re
import numpy as np
import networkx as nx
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from flask import Flask, request, render_template

app = Flask(__name__)


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# ── Text Processing ───────────────────────────────────────────────────────────

def read_article(text):
    sentences = text.split(". ")
    clean_sentences = []
    for sentence in sentences:
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        clean_sentences.append(sentence.split(" "))
    return clean_sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1

    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1], sentences[idx2], stop_words
            )
    return similarity_matrix


# ── Summarization ─────────────────────────────────────────────────────────────

def generate_summary(text, top_n=3):
    if not text.strip():
        return "Please enter some text."

    stop_words = stopwords.words('english')
    sentences = read_article(text)

    if len(sentences) <= top_n:
        return text

    similarity_matrix = gen_sim_matrix(sentences, stop_words)
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    summary_sentences = [" ".join(ranked_sentences[i][1]) for i in range(top_n)]
    return ". ".join(summary_sentences)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""

    if request.method == 'POST':
        if 'text' in request.form and request.form['text'].strip():
            summary = generate_summary(request.form['text'])

        elif 'pdf' in request.files:
            file = request.files['pdf']
            if file and file.filename != "":
                text = extract_text_from_pdf(file)
                summary = generate_summary(text)

    return render_template('index.html', summary=summary)


if __name__ == '__main__':
    app.run(debug=True)