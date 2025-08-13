import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load models and files ===
model = load_model("model/best_model.keras")
print("✅ Captioning model loaded")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load features dictionary (if required for your own dataset)
with open("model/features.pkl", "rb") as f:
    features_dict = pickle.load(f)

# Infer max length
max_length = 35  # Set this manually if not stored

# VGG16 feature extractor
vgg = VGG16()
vgg_model = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)
print("✅ VGG16 model ready")

# === Functions ===
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg_model.predict(img, verbose=0)
    return features

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    return ' '.join(final[1:-1])

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    if 'file' not in request.files:
        return "❌ No file part in request"
    
    file = request.files['file']
    if file.filename == '':
        return "❌ No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    features = extract_features(filepath)
    caption = generate_caption(model, tokenizer, features, max_length)

    return render_template('after.html', caption=caption, image=filename)

if __name__ == '__main__':
    app.run(debug=True)
