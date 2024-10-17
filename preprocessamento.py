from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(train_dir, test_dir, img_size=(64, 64), batch_size=32):
    """Configura o processamento e geração de dados de treino e teste."""
    # Data augmentation para o conjunto de treino
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Apenas normalização para o conjunto de teste
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Geradores de dados
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator
