import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from tensorflow.keras.models import Model #type:ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #type:ignore
from tensorflow.keras.optimizers import Adam #type:ignore
from tensorflow.keras.preprocessing import image #type:ignore
import shutil
import sys
import warnings
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set up Docker-agnostic paths (these paths should be mapped to the host system in Docker volumes)
base_dir = '/app/static'
employee_photos_dir = os.path.join(base_dir, 'employee_photos')
environment_photos_dir = os.path.join(base_dir, 'environment_photos')

# New directory for training
new_base_dir = '/app/training_classes'
os.makedirs(new_base_dir, exist_ok=True)

# Directory to save model outputs (make sure this is volume-mounted for persistence)
model_outputs_dir = '/app/model_outputs'
os.makedirs(model_outputs_dir, exist_ok=True)

# Function to copy photos from original structure to new folder
def copy_photos(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)

# Copy employee photos
for folder_name in os.listdir(employee_photos_dir):
    src_folder = os.path.join(employee_photos_dir, folder_name)
    dst_folder = os.path.join(new_base_dir, folder_name)
    copy_photos(src_folder, dst_folder)

# Copy environment photos
copy_photos(environment_photos_dir, os.path.join(new_base_dir, 'environment_photos'))

print(f"Data copied to {new_base_dir}")

# Set up data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load and preprocess the data from the new structure
train_generator = train_datagen.flow_from_directory(
    new_base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    new_base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Check sample counts
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Calculate steps per epoch
steps_per_epoch = max(1, np.ceil(train_generator.samples / train_generator.batch_size).astype(int))
validation_steps = max(1, np.ceil(validation_generator.samples / validation_generator.batch_size).astype(int))

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# Create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer with the correct number of classes
predictions = Dense(num_classes, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First, we only train the top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the total number of epochs you want to run
total_epochs = 4  # Set this to the number of epochs you want

# Train the model in a loop
for epoch in range(total_epochs):
    print(f"Epoch {epoch + 1}/{total_epochs}")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        epochs=1,  # Set to 1 for each iteration of the loop
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // validation_generator.batch_size)
    )

# Save the model
model_save_path = os.path.join(model_outputs_dir, 'model.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save the label_dict.json
label_dict = train_generator.class_indices  # Assuming this is created automatically
label_dict_path = os.path.join(model_outputs_dir, 'label_dict.json')
with open(label_dict_path, 'w') as f:
    json.dump(label_dict, f)
print(f"Label dictionary saved to {label_dict_path}")

# Function to predict the class of a single image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return train_generator.class_indices, preds[0]

# Test the model on a single image
test_image = '/app/static/test_images/diogo_2.jpg'
class_indices, prediction = predict_image(test_image)
class_names = {v: k for k, v in class_indices.items()}
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")
