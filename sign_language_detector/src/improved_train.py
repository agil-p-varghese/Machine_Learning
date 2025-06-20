from improved_preprocess import get_enhanced_data, augment_data
from improved_model import get_improved_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load and preprocess data
print("Loading and preprocessing data...")
X_train, X_val, y_train, y_val = get_enhanced_data("data/train.csv")

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Number of classes: {y_train.shape[1]}")

# Create improved model
model = get_improved_model()
model.summary()

# Enhanced callbacks
callbacks = [
    ModelCheckpoint("saved_models/best_model.h5", 
                   save_best_only=True, 
                   monitor='val_accuracy',
                   mode='max',
                   verbose=1),
    EarlyStopping(monitor='val_accuracy', 
                 patience=15, 
                 restore_best_weights=True,
                 verbose=1),
    ReduceLROnPlateau(monitor='val_loss', 
                     factor=0.5, 
                     patience=7, 
                     min_lr=0.00001,
                     verbose=1)
]

# Data augmentation
datagen = augment_data(X_train, y_train)

# Train the model with more epochs and data augmentation
print("Starting training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=50,  # Increased epochs
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)

print("Training completed!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")