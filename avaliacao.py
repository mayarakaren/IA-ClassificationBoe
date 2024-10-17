import csv
import os
import numpy as np

def salvar_metricas_e_previsoes(model, test_gen, history, class_indices, num_epocas=5):
    """Salva as métricas e previsões em um arquivo CSV."""
    # Salvando métricas das últimas épocas
    ultimas_epocas = {metric: history.history[metric][-num_epocas:] for metric in history.history}
    
    # Salvando métricas e previsões no CSV
    with open('metricas_finais.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(ultimas_epocas.keys())  # Cabeçalhos das métricas
        writer.writerows(zip(*ultimas_epocas.values()))  # Valores das métricas
        
        # Cabeçalhos para as previsões
        writer.writerow(['Imagem', 'Classe Verdadeira', 'Classe Prevista', 'Acurácia (%)'])
        
        for idx in range(len(test_gen.filenames)):
            img = test_gen[0][0][idx]  # Obtem a imagem
            true_label_idx = test_gen.labels[idx]  # Classe verdadeira
            pred = model.predict(np.expand_dims(img, axis=0))
            pred_class_idx = np.argmax(pred)
            pred_confidence = np.max(pred) * 100

            true_class = list(class_indices.keys())[list(class_indices.values()).index(true_label_idx)]
            pred_class = list(class_indices.keys())[list(class_indices.values()).index(pred_class_idx)]
            img_name = os.path.basename(test_gen.filenames[idx])

            # Salva a previsão
            writer.writerow([img_name, true_class, pred_class, f"{pred_confidence:.2f}%"])
