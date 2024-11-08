from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def merge_sort(files):
    """Aplica o merge sort para ordenar os arquivos."""
    if len(files) <= 1:
        return files
    
    mid = len(files) // 2
    left_half = merge_sort(files[:mid])
    right_half = merge_sort(files[mid:])

    sorted_files = []
    i = j = 0

    # Intercalar as duas metades ordenadas
    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            sorted_files.append(left_half[i])
            i += 1
        else:
            sorted_files.append(right_half[j])
            j += 1

    # Adicionar os elementos restantes
    sorted_files.extend(left_half[i:])
    sorted_files.extend(right_half[j:])

    return sorted_files

def preprocess_images(train_dir, test_dir, img_size=(64, 64), batch_size=32):
    """Configura o processamento e geração de dados de treino e teste com ordenação."""
    
    # Ordenar os arquivos no diretório de treino
    train_files = os.listdir(train_dir)
    sorted_train_files = merge_sort(train_files)
    
    # Ordenar os arquivos no diretório de teste
    test_files = os.listdir(test_dir)
    sorted_test_files = merge_sort(test_files)

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

    # Geradores de dados (o shuffle=False garante que as imagens sejam carregadas na ordem ordenada)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Para manter a ordem de carregamento
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Para manter a ordem de carregamento
    )

    return train_generator, test_generator
