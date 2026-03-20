from flask import Flask, render_template, request
import re
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
import numpy as np

app = Flask(__name__)


def read_article(text):
    sentences = text.split(". ")
    clean_sentences = []    

    for sentence in sentences:
        sentence = re.sub("[^a-zA-Z]", " ", sentence )
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
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=3):
   if not text.strip():
       return "Please enter some text."
   
   stop_words = stopwords.words('english')
   summarize_text = []      

   sentences = read_article(text)

   if len(sentences) <= top_n:
    return text

   sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
   sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
   scores = nx.pagerank(sentence_similarity_graph)

   ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

   for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))
   return ". ".join(summarize_text)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['text']
        summary = generate_summary(text)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)



 



 
