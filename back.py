from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your trained TensorFlow model
model = tf.keras.models.load_model('D:/2A/Project/Mines Greener 24/Degit_Recognizer_cnn.h5')

# Define endpoint for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define endpoint for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Receive the JSON data containing the grayscale image data
        data = request.get_json()
        grayscale_data = data['image']

        print([len(e) for e in grayscale_data])
        print(np.array(grayscale_data)*255)

        # flatten the 2D array to 1D
        image_array = np.array(grayscale_data).reshape(-1, 28, 28, 1)

        #use model to predict digit
        prediction = model.predict(image_array)

        #return the predicted digit
        predicted_digit = int(np.argmax(prediction))


        return jsonify({'predicted_digit': predicted_digit})


if __name__ == '__main__':
    app.run(debug=True)