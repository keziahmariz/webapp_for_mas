import os
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_toastr import Toastr
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, MultiHeadAttention, GlobalAveragePooling2D, Concatenate, Flatten, Dropout, LayerNormalization
from tensorflow.keras.models import load_model, Model
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

toastr = Toastr(app)
app.config['TOASTR_TIMEOUT'] = 10000  # Duration in milliseconds
app.config['TOASTR_POSITION_CLASS'] = 'toast-top-right'  # Position of the toast


# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
PREPROCESSED_IMG_FOLDER = "preprocessed_img"
PREPROCESSED_AUDIO_FOLDER = "preprocessed_audio"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
FUSED_NPY_FOLDER = "fused_npy"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREPROCESSED_IMG_FOLDER"] = PREPROCESSED_IMG_FOLDER
app.config["PREPROCESSED_AUDIO_FOLDER"] = PREPROCESSED_AUDIO_FOLDER
app.config["FUSED_NPY_FOLDER"] = FUSED_NPY_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_IMG_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_AUDIO_FOLDER, exist_ok=True)
os.makedirs(FUSED_NPY_FOLDER, exist_ok=True)

####---------------IMAGE-----------------####

# Register custom image processing functions
@tf.keras.utils.register_keras_serializable()
def preprocess_image(image):
    """
    Resize the image to (224, 224) and normalize pixel values to [0, 1].
    """
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

@tf.keras.utils.register_keras_serializable()
def expand_dims(x):
    """
    Add an extra dimension to the input tensor for attention layers.
    """
    return tf.expand_dims(x, axis=1)

@tf.keras.utils.register_keras_serializable()
def squeeze_dims(x):
    """
    Remove the extra dimension added for attention layers.
    """
    return tf.squeeze(x, axis=1)

@tf.keras.utils.register_keras_serializable()
class SaveEmbeddingModel(Model):
    """
    Custom Keras model to save embeddings during prediction.
    """

    def predict_and_save(self, image_path, output_path="image_embedding.npy"):
        """
        Preprocess image, pass it through the model, and save the embedding.

        Args:
            image_path: Path to the input image.
            output_path: Path to save the output embedding (default: 'image_embedding.npy').
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize image
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Generate the embedding
        embedding = self.predict(image_array)

        # Save the embedding to an .npy file
        np.save(output_path, embedding)
        print(f"Embedding saved to: {output_path}")
        return embedding

# Load the image preprocessing model
model_path = "models/image_pipeline_full_with_save_function.h5"  # Path to your preprocessing model
custom_objects = {
    "SaveEmbeddingModel": SaveEmbeddingModel,
    "preprocess_image": preprocess_image,
    "expand_dims": expand_dims,
    "squeeze_dims": squeeze_dims,
}
preprocess_model = load_model(model_path, custom_objects=custom_objects, compile=False)

def allowed_file(filename, allowed_extensions):
    """
    Check if a file's extension is allowed.
    """
    if "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()  # Extract file extension
    print(f"Validating extension: {ext}")  # Debug log
    return ext in allowed_extensions

# Helper function to preprocess the image
def preprocess_image(image_path):
    """
    Preprocess the uploaded image using the preprocessing model.

    Args:
        image_path: Path to the uploaded image.
    Returns:
        preprocessed_image_path: Path to the saved preprocessed embedding.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image has 3 color channels
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Generate the embedding using the preprocessing model
    image_embedding = preprocess_model.predict(image_array)

    # Save the preprocessed embedding as a .npy file
    # preprocessed_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_embedding.npy"
    preprocessed_image_path = os.path.join(app.config["PREPROCESSED_IMG_FOLDER"], "image_embedding.npy")
    np.save(preprocessed_image_path, image_embedding)
    
    return preprocessed_image_path




####----------------AUDIO-------------------####


# Register custom functions
@tf.keras.utils.register_keras_serializable()
def pad_or_truncate(features, fixed_timesteps=100):
    """
    Pad or truncate features to a fixed number of timesteps.
    """
    if features.shape[0] > fixed_timesteps:
        return features[:fixed_timesteps, :]
    elif features.shape[0] < fixed_timesteps:
        padding = np.zeros((fixed_timesteps - features.shape[0], features.shape[1]))
        return np.vstack([features, padding])
    return features

@tf.keras.utils.register_keras_serializable()
def expand_dims2(x):
    """
    Add an extra dimension to the input tensor for attention layers.
    """
    return tf.expand_dims(x, axis=1)

@tf.keras.utils.register_keras_serializable()
def squeeze_dims2(x):
    """
    Remove the extra dimension added for attention layers.
    """
    return tf.squeeze(x, axis=1)

@tf.keras.utils.register_keras_serializable()
class SaveEmbeddingModel2(Model):
    """
    Custom Keras model to save embeddings during prediction.
    """
    def predict_and_save(self, audio_path, output_path="audio_embedding.npy"):
        """
        Preprocess audio, pass it through the model, and save the embedding.

        Args:
            audio_path: Path to the input audio file.
            output_path: Path to save the output embedding (default: 'audio_embedding.npy').
        """
        # Load and preprocess the audio
        waveform, sr = librosa.load(audio_path, sr=16000)

        # Extract Wav2Vec2 features
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        wav2vec_model.eval()

        input_values = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        with torch.no_grad():
            features = wav2vec_model(input_values).last_hidden_state.squeeze(0).numpy()

        # Pad or truncate features
        features = pad_or_truncate(features)

        # Add batch dimension
        features = np.expand_dims(features, axis=0)

        # Generate the embedding
        embedding = self.predict(features)

        # Save the embedding to an .npy file
        np.save(output_path, embedding)
        print(f"Embedding saved to: {output_path}")
        return embedding

#Load audio preprocessing model
audio_model_path = "models/audio_pipeline_full_with_save_function.h5"  # Path to your preprocessing model
custom_objects = {
    "SaveEmbeddingModel": SaveEmbeddingModel2,
    "pad_or_truncate": pad_or_truncate,
    "expand_dims": expand_dims2,
    "squeeze_dims": squeeze_dims2,
}
audio_pre_model = load_model(audio_model_path, custom_objects=custom_objects, compile=False)


def preprocess_audio(audio_path):
   # Load and preprocess the audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Extract Wav2Vec2 features
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model.eval()

    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        features = wav2vec_model(input_values).last_hidden_state.squeeze(0).numpy()

    # Pad or truncate features
    features = pad_or_truncate(features)

    # Add batch dimension
    features = np.expand_dims(features, axis=0)

    # Generate the embedding using the audio pipeline model
    audio_embedding = audio_pre_model.predict(features)

    # Save the preprocessed embedding as a .npy file
    preprocessed_audio_path = os.path.join(app.config["PREPROCESSED_AUDIO_FOLDER"], "audio_embedding.npy")
    np.save(preprocessed_audio_path, audio_embedding)

    return preprocessed_audio_path




@app.route("/", methods=["GET"])
def index():
    """
    Render the main page and determine button states based on file existence.
    """
    # Check if the processed files exist
    image_processed = os.path.exists(os.path.join(app.config["PREPROCESSED_IMG_FOLDER"], "image_embedding.npy"))
    audio_processed = os.path.exists(os.path.join(app.config["PREPROCESSED_AUDIO_FOLDER"], "audio_embedding.npy"))
    files_processed = image_processed and audio_processed
    print(files_processed)

    return render_template("index.html", files_processed=files_processed)

@app.route("/process", methods=["POST"])
def process():
    """
    Handle the file upload, processing, and automatic fusion logic.
    """
    # Check if any file is in the request
    if "image_file" not in request.files and "audio_file" not in request.files:
        flash("No file selected. Please choose at least one file.", "error")
        return redirect(url_for("index"))

    image_embedding_path = None
    audio_embedding_path = None

    # Handle image file
    if "image_file" in request.files:
        image_file = request.files["image_file"]

        if image_file.filename == "":
            flash("No image file selected.", "error")
        elif not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash(f"Invalid image file. Allowed extensions: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}", "error")
        else:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(image_path)
            try:
                image_embedding_path = preprocess_image(image_path)
                flash(f"Image file '{image_file.filename}' uploaded and processed successfully.", "success")
            except Exception as e:
                flash(f"Error processing image file '{image_file.filename}': {str(e)}", "error")

    # Handle audio file
    if "audio_file" in request.files:
        audio_file = request.files["audio_file"]

        if audio_file.filename == "":
            flash("No audio file selected.", "error")
        elif not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            flash(f"Invalid audio file. Allowed extensions: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}", "error")
        else:
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)
            audio_file.save(audio_path)
            try:
                audio_embedding_path = preprocess_audio(audio_path)
                flash(f"Audio file '{audio_file.filename}' uploaded and processed successfully.", "success")
            except Exception as e:
                flash(f"Error processing audio file '{audio_file.filename}': {str(e)}", "error")

    # Check if both embeddings are saved and accessible
    if image_embedding_path and audio_embedding_path:
        try:
            # Load the embeddings
            image_embedding = np.load(image_embedding_path)
            audio_embedding = np.load(audio_embedding_path)

            # Ensure embeddings have compatible shapes
            if image_embedding.shape[0] != audio_embedding.shape[0]:
                flash("Mismatch in embedding shapes. Cannot fuse.", "error")
                return redirect(url_for("index"))

            # Fuse the embeddings by concatenating along the feature axis
            fused_embedding = np.concatenate((image_embedding, audio_embedding), axis=1)

            # Save the fused embedding
            fused_embedding_path = os.path.join(app.config["FUSED_NPY_FOLDER"], "fused_embedding.npy")
            np.save(fused_embedding_path, fused_embedding)

            flash(f"Embeddings fused successfully! Fused embedding saved at: {fused_embedding_path}", "success")
        except Exception as e:
            flash(f"Error during fusion: {str(e)}", "error")

    return redirect(url_for("index"))


    # Save the files
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], "image_embedding.npy")
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "audio_embedding.npy")
    image_file.save(image_path)
    audio_file.save(audio_path)

    # Preprocess the image
    try:
        preprocessed_image_path = preprocess_image(image_path)
        preprocessed_audio_path = preprocess_audio(audio_path)
    except Exception as e:
        flash(f"Error during preprocessing: {str(e)}", "error")
        return redirect(url_for("index"))

    # Flash success message and redirect
    flash(f"Files uploaded successfully: {image_file.filename} and {audio_file.filename}",
          "success",
          )
    flash(f"Image Preprocessed successfully at {preprocessed_image_path}",
          "success",
          )
    flash(f"Audio Preprocessed successfully at {preprocessed_audio_path}",
          "success",)
    return redirect(url_for("index"))




@app.route("/predict", methods=["POST"])
def predict():
    fused_path = os.path.join(app.config["FUSED_NPY_FOLDER"], "fused_embedding.npy")

    if not os.path.exists(fused_path):
        flash("Fused embedding not found. Please process files first.", "error")
        return redirect(url_for("index"))

    try:
        # Load fused embeddings
        fused_embeddings = np.load(fused_path, allow_pickle=True)
        print(f"Loaded Fused Embedding Shape: {fused_embeddings.shape}")

        # Select the first sample (or loop through samples if necessary)
        sample_index = 0
        sample_embedding = fused_embeddings[sample_index].reshape(1, -1)

        # Load the prediction model
        model_path = "models/final_esf_model.keras"
        model = load_model(model_path, compile=False)

        

        # Make prediction
        predictions = model.predict(sample_embedding)
        print(f"Model Prediction Output: {predictions}")

        # Determine predicted class and confidence
        #label_classes = ["0", "1", "2", "3", "4", "5", "6"]
        class_to_age_range = {
            0: "Teens (13-19)",
            1: "Twenties (20-29)",
            2: "Thirties (30-39)",
            3: "Fourties (40-49)",
            4: "Fifties (50-59)",
            5: "Sixties (60-69)",
            6: "Seventies+ (70+)"
            }

        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class_name = class_to_age_range[predicted_class_index]

        # Pass results to the template
        result_h1 = f"Age Group: {predicted_class_name}"
        analysis_h1 = f"{confidence:.4f}"

        return render_template(
            "index.html",
            files_processed=True,
            result_h1=result_h1,
            analysis_h1=analysis_h1
        )
    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "error")
        print(f"Prediction Error: {str(e)}")
        return redirect(url_for("index"))



@app.route("/reset", methods=["POST"])
def reset():
    """
    Handle the reset functionality.
    """
# Clear the uploads and preprocessed image folders
    for folder in [UPLOAD_FOLDER, PREPROCESSED_IMG_FOLDER, PREPROCESSED_AUDIO_FOLDER, FUSED_NPY_FOLDER]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

    flash("Files and state have been reset.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
