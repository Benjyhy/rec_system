import json
from flask import Flask, render_template
import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs

app = Flask(__name__)

with open('kmeans_rec.json', 'r') as f:
  kmeans_rec = json.load(f)

tf_rec = tf.saved_model.load('retrieval__model__cli_id')

class Model:
    def __init__(self, name, results):
        self.name = name
        self.results = results

    def __repr__(self):
        return '<Model %r>' % self.name

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/kmeans/<int:id>', methods=['GET'])
def kmeans(id):
    id = f"{id}"
    results = np.array([v for k,v in kmeans_rec[id].items()]).flatten().tolist()
    model = Model("kmeans", results)
    return render_template('recs.html', model = model)

@app.route('/tfr/<int:id>', methods=['GET'])
def tfr(id):
    encoding = 'utf-8'
    scores, rec = tf_rec(np.array([int(id)]))
    results = []
    for i in rec[0][:5].numpy().tolist():
        results.append(i.decode(encoding))
    model = Model("tfr", results)
    return render_template('recs.html', model = model)

if __name__ == '__main__':
    app.run(debug=True)