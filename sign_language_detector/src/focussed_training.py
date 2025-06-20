import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import *
from improved_preprocess import get_enhanced_data
from advanced_model import get_advanced_model, get_weighted_loss
import tensorflow as tf

def compute_class_weights(y_train):
    """Compute class weights to handle imbalanced data"""
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced',
                                       classes=np.unique(y_integers),
                                       y=y_integers)
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def create_hard_example_generator(X, y, problem_signs):
    """Generate more examples for difficult signs"""
    problem_indices = [ord(sign) - 65 for sign in problem_signs]
    
    # Find samples of problem signs
    y_classes = np.argmax(y, axis=1)
    hard_examples_X = []
    hard_examples_y = []
    
    for idx in problem_indices:
        sign_samples = X[y_classes == idx]
        sign_labels = y[y_classes == idx]
        
        # Generate additional variations for each sample
        for i in range(len(sign_samples)):
            original = sign_samples[i]
            label = sign_labels[i]
            
            # Create 3 additional variations
            variations = []
            
            # Rotation variations
            for angle in [-5, 0, 5]:
                # Simple rotation simulation by shifting pixels
                rotated = np.roll(original, angle, axis=1)
                variations.append(rotated)
            
            # Intensity variations
            for factor in [0.9, 1.0, 1.1]:
                intensity_var = np.clip(original * factor, 0, 1)
                variations.append(intensity_var)
            
            # Add variations to dataset
            for var in variations:
                hard_examples_X.append(var)
                hard_examples_y.append(label)
    
    return np.array(hard_examples_X), np.array(hard_examples_y)

def advanced_training_pipeline():
    """Advanced training pipeline with focus on problem signs"""
    
    print("Loading data...")
    X_train, X_val, y_train, y_val = get_enhanced_data("data/train.csv")
    
    # Identify problem signs (you'll get this from analysis)
    problem_signs = ['B', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U']  # Example
    
    print(f"Focusing on problem signs: {problem_signs}")
    
    # Generate additional hard examples
    hard_X, hard_y = create_hard_example_generator(X_train, y_train, problem_signs)
    
    # Combine with original data
    X_train_enhanced = np.concatenate([X_train, hard_X])
    y_train_enhanced = np.concatenate([y_train, hard_y])
    
    print(f"Enhanced training data: {X_train_enhanced.shape}")
    
    # Compute class weights
    class_weights = compute_class_weights(y_train_enhanced)
    print("Class weights computed")
    
    # Create advanced model
    model = get_advanced_model()
    print("Advanced model created")
    
    # Advanced callbacks
    callbacks = [
        ModelCheckpoint("saved_models/advanced_model.h5",
                       save_best_only=True,
                       monitor='val_accuracy',
                       mode='max',
                       verbose=1),
        
        EarlyStopping(monitor='val_accuracy',
                     patience=20,
                     restore_best_weights=True,
                     verbose=1),
        
        ReduceLROnPlateau(monitor='val_loss',
                         factor=0.3,
                         patience=8,
                         min_lr=0.00001,
                         verbose=1),
        
        # Cyclical Learning Rate
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.95 ** epoch)
        ),
        
        # Custom callback to track problem sign accuracy
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
            print(f"Epoch {epoch+1} - Val Acc: {logs['val_accuracy']:.4f}")
        )
    ]
    
    # Advanced data augmentation for problem signs
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=False,  # Don't flip for sign language
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Train with focus on problem signs
    print("Starting advanced training...")
    history = model.fit(
        datagen.flow(X_train_enhanced, y_train_enhanced, batch_size=32),
        steps_per_epoch=len(X_train_enhanced) // 32,
        epochs=75,  # More epochs
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,  # Use class weights
        verbose=1
    )
    
    return model, history

# Fine-tuning on specific problem signs
def fine_tune_problem_signs(base_model, problem_signs):
    """Fine-tune model specifically on problem signs"""
    
    # Load data
    X_train, X_val, y_train, y_val = get_enhanced_data("data/train.csv")
    
    # Filter to only problem signs
    problem_indices = [ord(sign) - 65 for sign in problem_signs]
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
    # Get samples of problem signs only
    problem_mask_train = np.isin(y_train_classes, problem_indices)
    problem_mask_val = np.isin(y_val_classes, problem_indices)
    
    X_problem_train = X_train[problem_mask_train]
    y_problem_train = y_train[problem_mask_train]
    X_problem_val = X_val[problem_mask_val]
    y_problem_val = y_val[problem_mask_val]
    
    print(f"Fine-tuning on {len(X_problem_train)} problem sign samples")
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    
    # Compile with lower learning rate
    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Fine-tune
    history = base_model.fit(
        X_problem_train, y_problem_train,
        validation_data=(X_problem_val, y_problem_val),
        epochs=20,
        batch_size=16,
        verbose=1
    )
    
    return base_model

if __name__ == "__main__":
    # Run advanced training
    model, history = advanced_training_pipeline()
    print("Advanced training completed!")
    
    # Optionally fine-tune on problem signs
    # model = fine_tune_problem_signs(model, problem_signs)
    # print("Fine-tuning completed!")