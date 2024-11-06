from preprocessamento import preprocess_images
from modelo import build_cnn, combine_models
from treino import train_model
import config

def main():
    # Pré-processamento de dados
    train_gen, test_gen = preprocess_images(config.train_dir, config.test_dir, config.img_size, config.batch_size)

    # Construção do modelo
    cnn_model = build_cnn(input_shape=(config.img_size[0], config.img_size[1], 3))
    model = combine_models(cnn_model)

    # Treinamento
    history = train_model(model, train_gen, test_gen, config.epochs)

if __name__ == '__main__':
    main()
