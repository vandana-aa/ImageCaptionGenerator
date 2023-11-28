from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to extract features from an image
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        return None
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

# Function to generate description
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Default image path
default_path = "/Users/vandana/Desktop/Image_CaptionGenerator/Flicker8k_Dataset/35506150_cbdb630f4f.jpg"
# Ask user for input
user_input = input("Enter the path of the image you want to predict (or type 'default' to use the default image): ")

# Use the default path if the user types 'default'
if user_input.lower() == 'default':
    img_path = default_path
else:
    img_path = user_input

# Model and tokenizer loading
max_length = 32
tokenizer = load(open("/Users/vandana/tokenizer.p","rb"))
model = load_model('/Users/vandana/models/model_1.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Feature extraction and prediction
# Feature extraction and prediction
photo = extract_features(img_path, xception_model)
if photo is not None:
    img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    # Remove 'start' and 'end' tokens from the description
    description = description.replace('start', '').replace('end', '').strip()

    # Display the image
    plt.imshow(img)

    # Display the caption below the image
    plt.text(0, img.size[1] + 40, description, color='black',
             backgroundcolor='white', wrap=True, fontsize=14, weight='bold')

    # Remove axis details
    plt.axis('off')

    # Show the plot
    plt.show()
