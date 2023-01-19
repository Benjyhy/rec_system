from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

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
    model = Model("kmeans", ["aaa", "bbb", "ccc"])
    return render_template('recs.html', model = model)

@app.route('/tfr/<int:id>', methods=['GET'])
def tfr(id):
    model = Model("tfr", ["eee", "fff", "ggg"])
    return render_template('recs.html', model = model)

if __name__ == '__main__':
    app.run(debug=True)