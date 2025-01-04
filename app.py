import os
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_toastr import Toastr

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

toastr = Toastr(app)
app.config['TOASTR_TIMEOUT'] = 5000  # Duration in milliseconds
app.config['TOASTR_POSITION_CLASS'] = 'toast-top-right'  # Position of the toast


# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename, allowed_extensions):
    """
    Check if a file's extension is allowed.
    """
    if "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()  # Extract file extension
    print(f"Validating extension: {ext}")  # Debug log
    return ext in allowed_extensions


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
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)
    image_file.save(image_path)
    audio_file.save(audio_path)

    # Flash success message and redirect
    flash(f"Files uploaded successfully: {image_file.filename} and {audio_file.filename}",
          "success",
          )
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    """
    Handle the reset functionality.
    """
    # Clear the uploads folder (optional)
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))

    flash("Files and state have been reset.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
