import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the transformer encoder layer
class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dense_proj = layers.Dense(embed_dim)

    def call(self, inputs, training=None):
        inputs_proj = self.dense_proj(inputs)
        attn_output = self.att(inputs_proj, inputs_proj)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_proj + attn_output)  # Residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual connection

# Define the transformer model using the updated TransformerEncoder class
def create_transformer_model(sequence_length, num_bones, features_per_bone, num_future_frames=1, num_heads=8, embed_dim=256, ff_dim=512, num_transformer_blocks=6):
    input_shape = (sequence_length, num_bones * features_per_bone)
    input_layer = layers.Input(shape=input_shape)
    
    # Project input to embed_dim
    x = layers.Dense(embed_dim)(input_layer)
    
    # Positional encoding
    pos_encoding = positional_encoding(sequence_length, embed_dim)
    x = x + pos_encoding
    
    # Apply multiple Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(num_heads=num_heads, embed_dim=embed_dim, ff_dim=ff_dim)(x)
    
    # Add LSTM layers
    x = layers.LSTM(512, return_sequences=True)(x)
    x = layers.LSTM(512, return_sequences=True)(x)
    
    # Dense layers for prediction
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # Output multiple future frames
    output = layers.Dense(num_bones * features_per_bone)(x[:, -1, :])  # Use only the last timestep
    output = layers.RepeatVector(num_future_frames)(output)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

if __name__ == "__main__":
    # Load the normalized data
    all_frames_normalized = np.load('all_frames_normalized.npy')
    bone_names = np.load('bone_names.npy')
    normalization_params = np.load('normalization_params.npy', allow_pickle=True).item()
    
    sequence_length = 10
    num_bones = len(bone_names)
    features_per_bone = all_frames_normalized.shape[1] // num_bones
    
    num_future_frames = 10  # Predict 5 future frames

    X = []
    y = []
    for i in range(len(all_frames_normalized) - sequence_length - num_future_frames + 1):
        X.append(all_frames_normalized[i:i + sequence_length])
        y.append(all_frames_normalized[i + sequence_length:i + sequence_length + num_future_frames])

    X = np.array(X)
    y = np.array(y)
    
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    model = create_transformer_model(sequence_length, num_bones, features_per_bone, num_future_frames=num_future_frames)
    model.summary()

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5)

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    ) 
    
    model.save('transformer_animation_predictor.h5')
    print("Model saved!")