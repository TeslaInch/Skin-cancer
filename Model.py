import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import numpy as np

def Transfername(namecode):
    if namecode == 0:
        return "Benign"
    else:
        return "Malignant"
def preprocess_image(image_path):
    im = Image.open(image_path).convert("RGB")
    im = im.resize((224, 224))  # resize with PIL
    img = np.array(im) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_output(img):
    loaded_model = tf.keras.models.load_model('my_model.h5', compile=False)

    pred_result =  loaded_model.predict(img)
    if pred_result[0][0] > 0.5:
        result = 1
    else:
        result = 0

    return result