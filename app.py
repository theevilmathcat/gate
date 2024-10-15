from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
    session,
    current_app,
)
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import os
import numpy as np
import base64
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from tensorflow.keras.models import Model #type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore
from tensorflow.keras.models import load_model #type:ignore
from sqlalchemy.dialects.postgresql import BYTEA #type:ignore
import tempfile
import shutil
import warnings
import logging
import sys
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from functools import lru_cache
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

app = Flask(__name__)
app.config["BASE_DIR"] = os.environ.get("BASE_DIR")
app.config["MODEL_PATH"] = os.environ.get("MODEL_PATH")
app.config["MODEL_OUTPUTS_DIR"] = "/app/model_outputs"
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
app.config["DATABASE_URL"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "postgresql://admin:admin@db/db3"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'], pool_size=5, max_overflow=10)


db = SQLAlchemy(app)
oauth = OAuth(app)
bcrypt = Bcrypt(app)

# Configure OAuth providers (replace with actual keys)
oauth.register(
    name="gmail",
    client_id=os.environ.get("GMAIL_CLIENT_ID"),
    client_secret=os.environ.get("GMAIL_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "redirect_uri": "http://localhost:5001/login/gmail/authorized",
    },
)

# New GATEClient model for software users
class GATEClient(db.Model):
    __tablename__ = "gate_users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    oauth_provider = db.Column(db.String(20))  # 'gmail', 'yandex', 'github', or None for local
    employees = db.relationship('Employee', backref='gate_client', lazy=True)
    
    # New columns for model storage
    trained_model = db.Column(BYTEA)
    label_dict = db.Column(db.JSON)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Save model and label dictionary to the user's record in the database
    def save_model(self, model, label_dict):
        # Get the path from environment variables (set via app config)
        model_path = app.config["MODEL_PATH"]
        
        # Ensure the directory exists
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Create a temporary file in the specified path
        with tempfile.NamedTemporaryFile(suffix='.keras', dir=model_path, delete=False) as temp_model_file:
            # Save the Keras model to this temporary file
            tf.keras.models.save_model(model, temp_model_file.name)
            
            # Read the saved model into bytes
            temp_model_file.seek(0)
            model_bytes = temp_model_file.read()
            self.trained_model = model_bytes

        # Save the label dictionary as JSON in the database
        self.label_dict = json.dumps(label_dict)
        
        # Commit the changes to the database
        db.session.commit()
        
        # Clear the loaded model cache for this client
        self.clear_loaded_model(self.id)

    _loaded_models = {}  # Class variable to store loaded models
    
    @classmethod
    @lru_cache(maxsize=None)
    def get_or_load_model(cls, client_id):
        if client_id not in cls._loaded_models:
            client = cls.query.get(client_id)
            if client and client.trained_model:
                with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
                    temp_model_file.write(client.trained_model)
                    temp_model_file.flush()
                    model = tf.keras.models.load_model(temp_model_file.name)
                label_dict = json.loads(client.label_dict)
                cls._loaded_models[client_id] = (model, label_dict)
            else:
                return None, None
        return cls._loaded_models[client_id]
    
    @classmethod
    def clear_loaded_model(cls, client_id):
        if client_id in cls._loaded_models:
            del cls._loaded_models[client_id]
        cls.get_or_load_model.cache_clear()

    def load_model(self):
        if self.trained_model:
            # Create a temporary file to save the model bytes
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
                temp_model_file.write(self.trained_model)  # Write the model bytes to the temp file
                temp_model_file.flush()  # Ensure all data is written to the file
                
                # Load the model from the temporary file
                model = tf.keras.models.load_model(temp_model_file.name)

            # Load the label dictionary from JSON
            label_dict = json.loads(self.label_dict)  # Assuming label_dict is stored as JSON in the database
            return model, label_dict
        return None, None

# Employee model
class Employee(db.Model):
    __tablename__ = "employee_data"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    job = db.Column(db.String(100), nullable=False)
    income = db.Column(db.Float, nullable=False)
    gate_client_id = db.Column(db.Integer, db.ForeignKey('gate_users.id'), nullable=False)


class EmployeePhoto(db.Model):
    __tablename__ = "employee_photos"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.Integer,
        db.ForeignKey("employee_data.id", ondelete="CASCADE"),
        nullable=False,
    )
    photo = db.Column(db.LargeBinary, nullable=False)


# New for User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "client_id" not in session:
            flash("Log in to access this page.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# Routes
@app.route("/connect_to_database")
@login_required
def connect_to_database():
    # Logic to reconnect to the database if needed
    # For example, if you have a function that verifies the connection:
    db.session.commit()  # Ensure any pending changes are saved
    flash("Connected to the database successfully.", "success")
    
    return redirect(url_for("index"))  # Redirect back to the index page

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form["email"]
        password = request.form["password"]
        
        user = db.session.query(GATEClient).filter(
            (GATEClient.username == identifier) | (GATEClient.email == identifier)
        ).first()

        if user and user.check_password(password):
            session.clear()  # Clear any existing session data
            session["client_id"] = user.id
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        
        flash("Invalid username/email or password", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Basic validation
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if db.session.query(GATEClient).filter_by(username=username).first():
            flash("Username already exists", "error")
        elif db.session.query(GATEClient).filter_by(email=email).first():
            flash("Email already registered", "error")
        else:
            new_client = GATEClient(username=username, email=email)
            new_client.set_password(password)
            db.session.add(new_client)
            db.session.commit()

            # Automatically log in the new user
            session["client_id"] = new_client.id
            flash(f"Registered and logged in successfully. \nWelcome {username}!", "success")
            return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login/gmail")
def login_gmail():
    # Check if force_login is set
    force_login = request.args.get("force_login", "false").lower() == "true"

    if force_login:
        # Clear any existing session
        session.clear()
        # Add a parameter to the authorization URL to force a new login
        extra_params = {"prompt": "select_account"}
    else:
        extra_params = {}

    redirect_uri = url_for("gmail_authorized", _external=True, _scheme="http")
    return oauth.gmail.authorize_redirect(redirect_uri, **extra_params)


@app.route("/login/gmail/authorized")
def gmail_authorized():
    try:
        token = oauth.gmail.authorize_access_token()
        resp = oauth.gmail.get("https://www.googleapis.com/oauth2/v2/userinfo")
        user_info = resp.json()

        # Check if the user is already logged in
        if "client_id" in session:
            current_client = db.session.get(GATEClient, session["client_id"])
            if current_client and current_client.email == user_info["email"]:
                flash("You are already logged in.", "info")
                return redirect(url_for("dashboard"))

        # Check if the user already exists in the database
        client = db.session.query(GATEClient).filter_by(email=user_info["email"]).first()
        if not client:
            client = GATEClient(
                username=user_info["email"],
                email=user_info["email"],
                oauth_provider="gmail",
            )
            db.session.add(client)
            db.session.commit()

        session["client_id"] = client.id
        flash("Logged in successfully via Gmail", "success")
        return redirect(url_for("dashboard"))
    except Exception as e:
        app.logger.error(f"Error in Gmail authorization: {str(e)}")
        flash("An error occurred during Gmail login. Please try again.", "error")
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# Add similar routes for Yandex and GitHub
@app.route("/dashboard")
@login_required
def dashboard():
    client = db.session.get(GATEClient, session["client_id"])
    return render_template("index.html", client=client)


@app.route("/")
@login_required
def index():
    client = db.session.get(GATEClient, session.get("client_id"))
    if client is None:
        flash("User not found. Please log in again.", "error")
        return redirect(url_for("login"))
    
    employees = db.session.query(Employee).filter_by(gate_client_id=client.id).all()
    return render_template("index.html", employees=employees)

@app.route("/add_employee", methods=["GET", "POST"])
@login_required
def add_employee():
    print("add_employee route called")
    if request.method == "POST":
        name = request.form["name"]
        age = int(request.form["age"])
        job = request.form["job"]
        income = float(request.form["income"])
        
        client = db.session.get(GATEClient, session["client_id"])

        # Save employee data
        new_employee = Employee(name=name, age=age, job=job, income=income, gate_client_id=client.id)
        db.session.add(new_employee)
        db.session.commit()

        # Check if photos were submitted
        if "photos" in request.form:
            photos = json.loads(request.form["photos"])

            # Save photos
            employee_folder_name = (
                f"{new_employee.id}-{new_employee.name.replace(' ', '_')}"
            )
            employee_folder_path = os.path.join(
                app.config["BASE_DIR"], employee_folder_name
            )
            os.makedirs(employee_folder_path, exist_ok=True)

            for i, photo_data in enumerate(photos):
                photo = base64.b64decode(photo_data.split(",")[1])
                photo_filename = f"{new_employee.id}_photo_{i + 1}.jpg"
                photo_path = os.path.join(employee_folder_path, photo_filename)

                with open(photo_path, "wb") as photo_file:
                    photo_file.write(photo)

                new_photo = EmployeePhoto(employee_id=new_employee.id, photo=photo)
                db.session.add(new_photo)

            db.session.commit()
        flash(f"{name} added to the database", "success")
        return (
            jsonify({"status": "success", "message": f"{name} added to the database"}),
            200,
        )

    return render_template("add_employee.html")


@app.route("/api/capture_photos", methods=["POST"])
@login_required
def capture_photos():
    data = request.get_json()
    photos = data.get("photos")
    employee_id = session.get("employee_id")  # Get the employee ID from the session

    if not employee_id:
        return jsonify({"error": "No employee ID found in session."}), 400

    # Fetch the employee from the database
    employee = db.session.query(Employee).filter_by(id=employee_id).first()

    if not employee:
        return jsonify({"error": "Employee not found."}), 400

    # Construct the folder path for the employee
    employee_folder_name = f"{employee.id}-{employee.name.replace(' ', '_')}"
    employee_folder_path = os.path.join(app.config["BASE_DIR"], employee_folder_name)

    # Ensure the employee directory exists
    os.makedirs(employee_folder_path, exist_ok=True)

    if photos:
        for i, photo_data in enumerate(photos):
            # Decode the photo from Base64
            photo = base64.b64decode(photo_data.split(",")[1])

            # Save the photo to the employee's folder
            photo_filename = f"{employee.id}_photo_{i + 1}.jpg"
            photo_path = os.path.join(employee_folder_path, photo_filename)

            # Write the photo to disk
            with open(photo_path, "wb") as photo_file:
                photo_file.write(photo)

            # save the photos in the database
            new_photo = EmployeePhoto(employee_id=employee_id, photo=photo)
            db.session.add(new_photo)

        db.session.commit()
        return jsonify({"message": "Photos captured and saved successfully!"}), 200
    return jsonify({"error": "No photos received."}), 400


@app.route("/finish_session", methods=["POST"])
@login_required
def finish_session():
    # Clear the employee ID from the session when the user finishes
    session.pop("employee_id", None)
    return jsonify({"message": "Employee session finished."}), 200


# Database connection function to get existing employee IDs
def get_existing_employee_ids():
    # Get the DATABASE_URL from the app configuration
    database_url = current_app.config["DATABASE_URL"]

    # Connect to the database using the DATABASE_URL
    connection = psycopg2.connect(database_url)
    cursor = connection.cursor()

    cursor.execute("SELECT id FROM employee_data;")
    employee_ids = [row[0] for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    return employee_ids

log_dir = "app/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "training_metrics.log"), level=logging.INFO)

@app.route("/train_model", methods=["POST"])
@login_required
def train_model():
    base_dir = "/app/model_inputs"
    employee_photos_dir = os.path.join(base_dir, "employee_photos")
    environment_photos_dir = os.path.join(base_dir, "environment_photos")
    new_base_dir = "app/training_classes"

    os.makedirs(new_base_dir, exist_ok=True)
    # Clear the training_classes directory before copying new data
    shutil.rmtree(new_base_dir)
    os.makedirs(new_base_dir)

    def copy_photos(src_dir, dst_dir):
        print(f"Copying from {src_dir} to {dst_dir}")
        os.makedirs(dst_dir, exist_ok=True)
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied file: {src_path} to {dst_path}")
            elif os.path.isdir(src_path):
                print(f"Found subdirectory: {src_path}")
                copy_photos(src_path, os.path.join(dst_dir, filename))

    # Get the current gate user's ID
    current_gate_user_id = session.get('client_id')
    if not current_gate_user_id:
        flash("You must be logged in to train the model.", "error")
        return redirect(url_for('login'))

    client = db.session.get(GATEClient, current_gate_user_id)
    if not client:
        flash("User not found.", "error")
        return redirect(url_for('login'))

    # Copy only current gate user's employee photos
    print(f"Copying employee photos from {employee_photos_dir}")
    current_employees = db.session.query(Employee).filter_by(gate_client_id=current_gate_user_id).all()
    for employee in current_employees:
        folder_name = f"{employee.id}-{employee.name.replace(' ', '_')}"
        src_folder = os.path.join(employee_photos_dir, folder_name)
        dst_folder = os.path.join(new_base_dir, folder_name)
        if os.path.isdir(src_folder):
            copy_photos(src_folder, dst_folder)
        else:
            print(f"No photos found for employee: {employee.name}")

    # Copy environment photos
    copy_photos(environment_photos_dir, os.path.join(new_base_dir, "environment_photos"))

    print(f"Data copied to {new_base_dir}")

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        new_base_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        new_base_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")

    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_epochs = 4

    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
            epochs=1,
            validation_data=validation_generator,
            validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
        )
        
        # After each epoch, evaluate the model on the validation set
        val_preds = model.predict(validation_generator)
        val_labels = validation_generator.classes
        val_preds_classes = val_preds.argmax(axis=1)

        # Compute metrics
        accuracy = accuracy_score(val_labels, val_preds_classes)
        precision = precision_score(val_labels, val_preds_classes, average='weighted')
        recall = recall_score(val_labels, val_preds_classes, average='weighted')
        f1 = f1_score(val_labels, val_preds_classes, average='weighted')

        # Log metrics
        logging.info(f"Epoch {epoch + 1}:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info("\n")

    # After training the model, save label dictionary to the database
    current_gate_user_id = session.get('client_id')
    client = db.session.get(GATEClient, current_gate_user_id)
    if not client:
        flash("User not found.", "error")
        return redirect(url_for('login'))

    # Save model and label dictionary to the user's record in the database
    label_dict = train_generator.class_indices
    client.save_model(model, label_dict)

    flash("Model trained and saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/recognize_employee")
@login_required
def recognize_employee():
    current_gate_user_id = session.get('client_id')
    if not current_gate_user_id:
        flash("You must be logged in to use this feature.", "error")
        return redirect(url_for('login'))

    client = db.session.get(GATEClient, current_gate_user_id)
    if not client:
        flash("User not found.", "error")
        return redirect(url_for('login'))

    model_trained = client.trained_model is not None

    return render_template("recognize_employee.html", model_trained=model_trained)

# Define predict function outside of the endpoint
def predict(model, image):
    return model(image, training=False)

@app.route("/api/recognize", methods=["POST"])
@login_required
def recognize():
    app.logger.info(f"Request files: {request.files}, Request form: {request.form}")
    current_gate_user_id = session.get('client_id')
    if not current_gate_user_id:
        return jsonify({"error": "You must be logged in to use this feature."}), 401

    model, label_dict = GATEClient.get_or_load_model(current_gate_user_id)
    if model is None or label_dict is None:
        return jsonify({"error": "Model is not available. Please train the model first."}), 400

    # Expecting the image to be sent via Form Data
    if 'image' not in request.files:
        app.logger.error("No image file provided.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    if file.filename == '':
        app.logger.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the image file
        image_data = file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions[0])
        class_names = {v: k for k, v in label_dict.items()}
        predicted_class = class_names[predicted_class_index]
        confidence = float(np.max(predictions[0]))

        if predicted_class == "environment_photos":
            return jsonify({
                "recognized": False,
                "name": "Unrecognized - Access Denied",
                "confidence": confidence,
            })
        else:
            employee_name = predicted_class.split("-")[1].replace("_", " ")
            return jsonify({
                "recognized": True,
                "name": f"Welcome {employee_name} - Access Granted",
                "confidence": confidence,
            })

    return jsonify({"error": "Failed to process the image"}), 400

    
@app.route("/remove_employee", methods=["POST"])
@login_required
def remove_employee():
    employee_id = request.form.get("employee_id", "").strip()

    if not employee_id:
        flash("Employee ID cannot be empty!", "error")
        return redirect(url_for("index"))

    try:
        employee_id = int(employee_id)
    except ValueError:
        flash("Invalid Employee ID!", "error")
        return redirect(url_for("index"))

    employee = db.session.query(Employee).filter_by(id=employee_id).first()

    if employee:
        try:
            db.session.query(EmployeePhoto).filter_by(employee_id=employee.id).delete()

            folder_name = f"{employee.id}-{employee.name.replace(' ', '_')}"
            folder_path = os.path.join("/app/training_classes", folder_name)

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted folder for employee {employee.name}: {folder_path}")
            else:
                print(f"No folder found for employee {employee.name}.")

            db.session.delete(employee)
            db.session.commit()

            flash(f"Employee {employee_id} and their photos removed successfully!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred while removing the employee: {str(e)}", "error")
    else:
        flash(f"Employee {employee_id} not found!", "error")

    return redirect(url_for("index"))

# Add a new route to clear the model
@app.route("/clear_model", methods=["POST"])
@login_required
def clear_model():
    current_gate_user_id = session.get('client_id')
    if not current_gate_user_id:
        flash("You must be logged in to clear the model.", "error")
        return redirect(url_for('login'))

    client = db.session.get(GATEClient, current_gate_user_id)
    if not client:
        flash("User not found.", "error")
        return redirect(url_for('login'))

    client.trained_model = None
    client.label_dict = None
    db.session.commit()

    GATEClient.clear_loaded_model(current_gate_user_id)  # Clear the cached model

    flash("Model cleared successfully.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=False)