from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Carregar o modelo treinado
try:
    model = load_model('bovino_classification_model.keras')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit(1)

# Classes do modelo
class_labels = ['Dermatite Nodular', 'Berne', 'Saudável']

def preprocess_image(img_path):
    """Carrega e pré-processa a imagem antes de passar para o modelo."""
    img = image.load_img(img_path, target_size=(64, 64))  # Ajustar o tamanho conforme o que foi usado no treinamento
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão extra
    img_array /= 255.0  # Normalizar a imagem para os valores entre 0 e 1
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada!"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Nenhuma imagem selecionada!"}), 400
    
    # Salvar a imagem enviada
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Pré-processar a imagem
    img_array = preprocess_image(file_path)

    # Fazer a predição
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # A confiança da previsão

    # Pegar o nome da classe prevista
    predicted_class = class_labels[predicted_class_idx]

    # Remover a imagem do servidor após a predição
    os.remove(file_path)

    # Retornar o resultado da predição e a confiança
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    # Criar a pasta 'uploads' se não existir
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0', port=5000)
