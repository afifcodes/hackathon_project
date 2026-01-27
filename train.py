import os
import math
import shutil
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-5

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

    # Data Generators (augmentation only for training)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes: {list(train_generator.class_indices.keys())}")

    # Check for existing checkpoint to resume
    if os.path.exists('models/model_best.h5'):
        print("Found checkpoint 'models/model_best.h5'. Resuming training...")
        model = tf.keras.models.load_model('models/model_best.h5')
        # Re-compile to reset optimizer and avoid "Unknown variable" errors from loaded state
        print("Re-compiling loaded model...")
        model.compile(optimizer=Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
    else:
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
    callbacks = [
        ModelCheckpoint('models/model_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
    ]

    history = None
    history_ft = None
    
    try:
        print("Starting Phase 1 Training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
            validation_data=validation_generator,
            validation_steps=math.ceil(validation_generator.samples / BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=callbacks
        )

        # Fine-tune top layers for a small accuracy boost
        print("Starting Phase 2 (Fine-tuning)...")
        history_ft = None
        # Unfreeze top layers - flexible for loaded models
        model.trainable = True
        # Freeze all except top 40
        for layer in model.layers[:-40]:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history_ft = model.fit(
            train_generator,
            steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
            validation_data=validation_generator,
            validation_steps=math.ceil(validation_generator.samples / BATCH_SIZE),
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nTraining crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save model
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Backup existing
        if os.path.exists('models/model.h5'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            shutil.copy2('models/model.h5', f'models/model_backup_{ts}.h5')

        # Prefer best checkpoint if available
        if os.path.exists('models/model_best.h5'):
            print("Loading best checkpoint for final save...")
            try:
                model = tf.keras.models.load_model('models/model_best.h5')
            except:
                print("Could not load model_best.h5, using current model state.")
        
        model.save('models/model.h5')
        print("Model saved as models/model.h5")
        
        # Save class indices
        with open('models/class_indices.txt', 'w') as f:
            f.write(str(train_generator.class_indices))

        # Save training history
        if history and history.history:
            from logger import save_training_results
            combined_history = history.history
            if history_ft and history_ft.history:
                for k, v in history_ft.history.items():
                    combined_history.setdefault(k, [])
                    combined_history[k].extend(v)
            save_training_results(combined_history, EPOCHS + FINE_TUNE_EPOCHS, num_classes)

if __name__ == "__main__":
    train_model()
