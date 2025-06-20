import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class AttentionLayer(Layer):
    """Simple spatial attention mechanism"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention weights
        attention_weights = tf.nn.softmax(tf.tensordot(x, self.W, axes=1), axis=1)
        # Apply attention
        attended_x = x * attention_weights
        return attended_x

def get_advanced_model():
    """Advanced model with residual connections and attention"""
    inputs = Input(shape=(28, 28, 1))
    
    # Initial conv layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    
    # Residual Block 1
    residual1 = x
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual1])  # Skip connection
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)
    
    # Residual Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    residual2 = x
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual2])  # Skip connection
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)
    
    # Attention mechanism
    x = AttentionLayer()(x)
    
    # Residual Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    residual3 = x
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual3])  # Skip connection
    x = Dropout(0.3)(x)
    
    # Global Average Pooling instead of Flatten
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with residual connection
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    dense_residual = x
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Add()([x, dense_residual])  # Dense skip connection
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(26, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use cyclical learning rate
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps=1000)
    
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 'top_3_accuracy'])
    
    return model

def get_ensemble_model():
    """Create ensemble of different architectures"""
    # Model 1: CNN with attention
    model1 = get_advanced_model()
    
    # Model 2: Different architecture
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(26, activation='softmax')(x)
    model2 = Model(inputs=inputs, outputs=outputs)
    
    # Ensemble approach
    input_layer = Input(shape=(28, 28, 1))
    pred1 = model1(input_layer)
    pred2 = model2(input_layer)
    
    # Average predictions
    averaged = Average()([pred1, pred2])
    ensemble_model = Model(inputs=input_layer, outputs=averaged)
    
    ensemble_model.compile(optimizer=Adam(learning_rate=0.0005),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    return ensemble_model

# Focus training on difficult signs
def get_weighted_loss(class_weights):
    """Create weighted loss function for imbalanced classes"""
    def weighted_categorical_crossentropy(y_true, y_pred):
        # Calculate weighted loss
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        unweighted_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_losses = unweighted_losses * weights
        return weighted_losses
    
    return weighted_categorical_crossentropy