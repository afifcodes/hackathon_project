import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Config
DATA_DIR = 'dataset'
# IMG_SIZE is 224x224 as it is the standard input size for MobileNetV2 
# trained on ImageNet, ensuring optimal feature extraction.
IMG_SIZE = (224, 224)
# Batch size of 32 provides a good balance between training stability 
# and memory efficiency, allowing the model to generalize effectively.
BATCH_SIZE = 32
# Epochs limited to 8-12 for hackathon speed while ensuring 
# sufficient convergence for a functional prototype.
EPOCHS = 12

def get_dataset_dir():
    # If the user has structured it as dataset/<plant_name>/<classes>, find the plant name
    if os.path.exists(DATA_DIR):
        subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if len(subdirs) == 1:
            print(f"Found dataset for plant: {subdirs[0]}")
            return os.path.join(DATA_DIR, subdirs[0])
        elif len(subdirs) > 1:
            print(f"Multiple plants found: {subdirs}. Using the first one: {subdirs[0]}")
            return os.path.join(DATA_DIR, subdirs[0])
        else:
            # Fallback if classes are directly in dataset/ (unlikely per spec but possible)
            return DATA_DIR
    return DATA_DIR

def train_model():
    dataset_path = get_dataset_dir()
    print(f"Training on dataset at: {dataset_path}")

    # Data Generators (with augmentation for training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes: {list(train_generator.class_indices.keys())}")

    # Base Model: MobileNetV2
    # MobileNetV2 is chosen for its efficiency and low latency, 
    # making it suitable for future mobile/edge deployment in the field.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base logic (Transfer Learning)
    # By freezing base layers pretrained on ImageNet, we retain the general image recognition 
    # capabilities (edges, textures) and only train the top layers for our specific task.
    base_model.trainable = False

    # Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    # Dense layer with softmax activation is used for multi-class classification
    # to provide a probability distribution over the predicted classes.
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # The Adam optimizer is chosen for its adaptive learning rate capabilities, 
    # making it robust and efficient for training neural networks.
    # Categorical Crossentropy is the standard loss function for multi-class 
    # problems where each image belongs to exactly one category.
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/model.h5')
    print("Model saved as models/model.h5")
    
    # Save class indices
    with open('models/class_indices.txt', 'w') as f:
        f.write(str(train_generator.class_indices))

if __name__ == "__main__":
    train_model()
