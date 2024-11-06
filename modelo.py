from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def thresholding(x, threshold=0.5):
    """Aplica limiarização para tornar as imagens binárias."""
    x = tf.clip_by_value(x, 0, 1)
    return tf.where(x > threshold, 1.0, 0.0)

@tf.keras.utils.register_keras_serializable()
def crop_and_roi(x):
    """Aplica um recorte central para focar em regiões de interesse."""
    return tf.image.central_crop(x, central_fraction=0.7)
    
# Adicione 'crop_and_roi' aos objetos personalizados
get_custom_objects().update({'crop_and_roi': crop_and_roi})

def build_cnn(input_shape):
    """Constrói a parte CNN do modelo."""
    inputs = layers.Input(shape=input_shape)
    
    # Aplicação de recorte e limiarização
    x = layers.Lambda(crop_and_roi)(inputs)
    x = layers.Lambda(thresholding)(x)
    
    # Camadas convolucionais com regularização L2
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)  # Achatar para alimentar a MLP
    
    return models.Model(inputs=inputs, outputs=x)

def combine_models(cnn_model):
    """Combina a CNN com a MLP."""
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(cnn_model.output)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(3, activation='softmax')(x)

    final_model = models.Model(inputs=cnn_model.input, outputs=output)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return final_model
