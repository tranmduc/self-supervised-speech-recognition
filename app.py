from flask import Flask, render_template, request, redirect, flash, url_for
import main
import urllib.request
from werkzeug.utils import secure_filename
from main import getPrediction
import os

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            label = getPrediction(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            flash(label)
            return redirect('/')

if __name__ == "__main__":
    app.run()