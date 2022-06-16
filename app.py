import os

import numpy as np
import torch
import pickle
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from source.image_data_reader import create_vocabulary
from source.models.resnet_scene import ResnetScene
from source.models.resnet_object import ResnetObject
from source.experiment import predict

UPLOAD_FOLDER = './static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://pprxpkztttmszw:040c786b916c85f71712f49c3710eb2f0dd4dab1bb75d16756d7c3d5aecfa5da@ec2-52-72-99-110.compute-1.amazonaws.com:5432/df4i2dsb72s0p9'
db = SQLAlchemy(app)


class Hashtag(db.Model):
    hashtag = db.Column(db.String(200), primary_key=True)
    ratings = db.Column(db.Float, nullable=False)
    repeat = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Hashtag {self.hashtag}>'


class Rating(db.Model):
    index = db.Column(db.Integer, primary_key=True)
    ratings = db.Column(db.Float, nullable=False)
    repeat = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Rating {self.index}>'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "MarkerClustered"
one_shot = True
folder_dir = './source/HARRISON/'
file = 'new_tag_list.txt'

vec_matrix = np.load("./source/hashtagembed.npz")['wordvec']
word_vec_dict = {idx: vec for idx, vec in enumerate(vec_matrix)}
_, _, vocabulary = create_vocabulary(os.path.join(folder_dir, file), {'threshold': 4})

word2idx = {k: v for v, k in enumerate(vocabulary, 1)}
word2idx['UNK'] = 0

idx2word = {v: k for v, k in enumerate(vocabulary, 1)}
idx2word[0] = 'UNK'
vocabulary = vocabulary + ['UNK']

resnet_object = ResnetObject(len(vocabulary) if one_shot else 500)
resnet_scene = ResnetScene(len(vocabulary) if one_shot else 500)
resnet_object.load_state_dict(torch.load('./source/save_dir/' + 'best_val_resnet_object.pth',
                                         map_location=torch.device('cpu')))
resnet_scene.load_state_dict(torch.load('./source/save_dir/' + 'best_val_resnet_scene.pth',
                                        map_location=torch.device('cpu')))

with open('./saved_mean_10_v2.pkl', 'rb') as f:
    mean_pickle = pickle.load(f)

with open('./saved_standard_10_v2.pkl', 'rb') as f:
    standard_pickle = pickle.load(f)

pickles = (mean_pickle, standard_pickle)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.template_filter()
def confidence_format(value):
    value = float(value)
    return "{:4.2f}%".format(value)


@app.template_filter()
def average_format(value):
    value = float(value)
    return "{:1.2f} / 5".format(value)


@app.template_filter()
def convert_int_to_str(value):
    value = int(value)
    return "{}".format(value)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/hashtag_rate', methods=['POST'])
def hashtag_rate():
    if request.method == 'POST':
        data = request.get_json(force=True)
        rating = data['rating']
        hashtag = data['hashtag']
        rate_index = data['index']
        filename = data['filename']

        try:
            hashtag_entry = Hashtag.query.get_or_404(hashtag)
            hashtag_entry.repeat = hashtag_entry.repeat + 1
            hashtag_entry.ratings = hashtag_entry.ratings + rating
            try:
                db.session.commit()
            except:
                return 'There was an issue about rating the selected hashtag'

        except:
            try:
                new_hashtag = Hashtag(hashtag=hashtag, ratings=rating, repeat=1)
                db.session.add(new_hashtag)
                db.session.commit()
            except:
                return 'There was an issue about rating the selected hashtag'

        try:
            rating_entry = Rating.query.get_or_404(rate_index)
            rating_entry.repeat = rating_entry.repeat + 1
            rating_entry.ratings = rating_entry.ratings + rating
            try:
                db.session.commit()
            except:
                return 'There was an issue about rating the selected hashtag'
        except:
            try:
                new_entry = Rating(index=rate_index, ratings=rating, repeat=1)
                db.session.add(new_entry)
                db.session.commit()
            except:
                return 'There was an issue about rating the selected hashtag'

        return f'Hashtag {hashtag} rated with {rating} stars.'


@app.route('/result/', methods=['POST'])
def upload_image(filename=None):
    if filename is None:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            hashtags = predict(resnet_object, resnet_scene, idx2word, pickles, {"one_shot": one_shot, "image_path": path})
            # hashtags = predict(resnet_object, idx2word, confidences_pickle,
            #                    {"one_shot": one_shot, "image_path": path})
            flash('Image successfully uploaded and displayed below')

            first, second = [], []

            for i in range(len(hashtags)):
                hashtag = hashtags[i][0]
                value = hashtags[i][1]
                try:
                    hashtag_entry = Hashtag.query.get_or_404(hashtag)
                    rating = hashtag_entry.ratings / hashtag_entry.repeat
                except:
                    rating = 0

                if i % 2 == 0:
                    first.append([hashtag, value, rating])

                else:
                    second.append([hashtag, value, rating])

            return render_template('index.html', filename=filename, first=first,
                                   second=second, len_f=len(first), len_s=len(second))
        else:
            flash('Only jpg format is allowed.')
            return redirect(request.url)
    else:
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        hashtags = predict(resnet_object, resnet_scene, idx2word, pickles, {"one_shot": one_shot, "image_path": path})
        flash('Image successfully uploaded and displayed below')

        first, second = [], []
        for i in range(len(hashtags)):

            hashtag = hashtags[i][0]
            value = hashtags[i][1]
            try:
                hashtag_entry = Hashtag.query.get_or_404(hashtag)
                rating = hashtag_entry.ratings / hashtag_entry.repeat
            except:
                rating = 0

            if i % 2 == 0:
                first.append([hashtag, value, rating])

            else:
                second.append([hashtag, value, rating])

        return render_template('index.html', filename=filename, first=first,
                               second=second, len_f=len(first), len_s=len(second))


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
