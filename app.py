import os
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_toastr import Toastr
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, MultiHeadAttention, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import load_model, Model
from PIL import Image

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

toastr = Toastr(app)
app.config['TOASTR_TIMEOUT'] = 5000  # Duration in milliseconds
app.config['TOASTR_POSITION_CLASS'] = 'toast-top-right'  # Position of the toast


# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
PREPROCESSED_IMG_FOLDER = "preprocessed_img"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREPROCESSED_IMG_FOLDER"] = PREPROCESSED_IMG_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_IMG_FOLDER, exist_ok=True)

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
    
@app.route("/", methods=["GET"])
def index():
    """
    Render the main page.
    """
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Handle the file upload and processing logic.
    """
    # Validate files in the request
    if "image_file" not in request.files or "audio_file" not in request.files:
        flash("No file selected. Please choose both files.", "error")
        return redirect(url_for("index"))

    image_file = request.files["image_file"]
    audio_file = request.files["audio_file"]

    # Check if files are valid
    if image_file.filename == "" or audio_file.filename == "":
        flash("No file selected. Please choose both files.")
        return redirect(url_for("index"))

    # Validate image file
    if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        flash(
            f"Invalid image file. Allowed extensions: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
            "error",
        )
        return redirect(url_for("index"))

    # Validate audio file
    if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
        flash(
            f"Invalid audio file. Allowed extensions: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
            "error",
        )
        return redirect(url_for("index"))

    # Save the files
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], "image_embedding.npy")
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "audio_embedding.npy")
    image_file.save(image_path)
    audio_file.save(audio_path)

    # Preprocess the image
    try:
        preprocessed_image_path = preprocess_image(image_path)
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
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    """
    Handle the reset functionality.
    """
# Clear the uploads and preprocessed image folders
    for folder in [UPLOAD_FOLDER, PREPROCESSED_IMG_FOLDER]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

    flash("Files and state have been reset.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
