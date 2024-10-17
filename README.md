# Projeto de Classificação de Lesões Dermatológicas em Bovinos

Este projeto tem como objetivo treinar um modelo de rede neural convolucional (CNN), combinado com um Perceptron Multicamadas (MLP), para classificar lesões dermatológicas em bovinos em três categorias: **Dermatite**, **Berne** e **Saudável**. Além disso, há uma API desenvolvida em Flask que permite fazer predições usando o modelo treinado.

## Estrutura do Projeto

O projeto está dividido em diferentes arquivos para facilitar a organização:

- `main.py`: Arquivo principal para executar o treinamento do modelo CNN + MLP.
- `treino.py`: Implementa as funções e callbacks necessários para o treinamento.
- `modelo.py`: Contém as funções para construir as partes da CNN e MLP.
- `preprocessamento.py`: Funções responsáveis pelo pré-processamento de imagens e data augmentation.
- `api.py`: API Flask para predições com o modelo treinado.
- `requirements.txt`: Lista de dependências do projeto.
- `README.md`: Instruções de uso do projeto (este arquivo).

## Requisitos

Antes de executar o projeto, certifique-se de ter instalado as dependências necessárias. Para isso, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

### Principais Dependências:

- **TensorFlow**: Para criar e treinar o modelo de rede neural.
- **Flask**: Para criar a API de predições.
- **Pillow**: Para manipulação de imagens.
- **numpy**: Para processamento de dados e operações numéricas.
- **csv**: Para salvar os resultados em arquivos CSV.

## Passo a Passo para Executar o Treinamento

### 1. Estrutura dos Dados

Os dados devem estar organizados da seguinte forma:

```
image/
├── train/
│   ├── berne/
│   ├── dermatite/
│   └── saudavel/
└── test/
    ├── berne/
    ├── dermatite/
    └── saudavel/
```

- A pasta `train/` contém as imagens de treinamento, enquanto a pasta `test/` contém as imagens de teste.
- Cada subpasta representa uma classe (`berne`, `dermatite`, `saudavel`).

### 2. Configuração do Treinamento

O treinamento do modelo é iniciado através do arquivo `main.py`. Ele utiliza a arquitetura CNN + MLP para classificação e inclui técnicas como data augmentation e regularização L2 para evitar overfitting.

#### Principais etapas do treinamento:

- **Data Augmentation**: Aumenta a variabilidade do conjunto de dados aplicando transformações como rotação e espelhamento nas imagens de treino.
- **CNN e MLP**: A CNN é responsável por extrair características das imagens, enquanto a MLP faz a classificação final.
- **Callbacks**: Inclui Early Stopping (interrompe o treinamento ao detectar estagnação) e Model Checkpoint (salva o melhor modelo durante o treinamento).
- **Métricas**: Salva as métricas do desempenho do modelo no arquivo `metricas_finais.csv`.

### Execução do Treinamento

Para iniciar o treinamento, execute o seguinte comando:

```bash
python main.py
```

Isso iniciará o treinamento usando os dados de `image/train` para treinar o modelo, e os dados de `image/test` para validar. O modelo treinado será salvo no arquivo `bovino_classification_model.keras`, e as métricas serão salvas no arquivo `metricas_finais.csv`.

### 3. Avaliação do Modelo

Após o treinamento, o modelo será avaliado nos dados de teste, e as previsões para algumas imagens selecionadas serão salvas em um arquivo CSV, junto com a classe verdadeira e a confiança da predição.

---

## API para Predições

O projeto inclui uma API desenvolvida com Flask que permite realizar predições usando o modelo treinado.

### Funcionamento

- A API recebe uma imagem de lesão bovina enviada pelo usuário.
- A imagem é pré-processada e passada pelo modelo para gerar uma predição.
- A API retorna a classe prevista (Dermatite, Berne ou Saudável) e a confiança dessa predição.

### Execução da API

Para iniciar a API, execute o arquivo `api.py`:

```bash
python api.py
```

Isso iniciará um servidor local Flask acessível em `http://127.0.0.1:5000`.

### Endpoints da API

A API possui um único endpoint para predições:

#### `POST /predict`

Este endpoint aceita uma imagem enviada como `multipart/form-data`. O arquivo deve ser enviado no campo `file`.

Exemplo de requisição usando `curl`:

```bash
curl -X POST -F file=@caminho/para/imagem.jpg http://127.0.0.1:5000/predict
```

A resposta será um JSON com a classe prevista e a confiança da predição:

```json
{
  "classe": "Dermatite",
  "confianca": "95.45%"
}
```

### Estrutura do Código da API

- **Carregamento do Modelo**: O modelo treinado é carregado assim que a API é iniciada.
- **Pré-processamento da Imagem**: A imagem enviada é redimensionada para 64x64 pixels, normalizada e ajustada para o formato de entrada esperado pelo modelo.
- **Predição**: A imagem pré-processada é passada pelo modelo, e a classe com maior confiança é retornada.
- **Resposta da API**: A API retorna a classe prevista e a porcentagem de confiança da predição.

---

## Explicações Detalhadas dos Códigos

### `main.py`

Este é o arquivo principal do projeto. Ele coordena o processo de treinamento, chamando funções de pré-processamento e construção de modelo, além de definir os callbacks e métricas. A execução deste arquivo inicia o treinamento da CNN + MLP, e as métricas são salvas em CSV.

### `modelo.py`

Este arquivo contém funções para construir a arquitetura do modelo:

- **`build_cnn()`**: Constrói a parte convolucional (CNN) do modelo, responsável por extrair características das imagens.
- **`build_mlp()`**: Constrói a parte MLP (Perceptron Multicamadas) usada para a classificação final.
- **`combine_models()`**: Combina a CNN e o MLP em um único modelo integrado.

### `preprocessamento.py`

Este arquivo cuida do pré-processamento das imagens:

- **`thresholding()`**: Aplica limiarização nas imagens, tornando-as binárias.
- **`crop_and_roi()`**: Executa um recorte central para focar nas regiões de interesse da imagem.

### `api.py`

Este arquivo implementa a API usando Flask. Sua função principal, `predict`, recebe uma imagem, aplica as funções de pré-processamento, passa pelo modelo e retorna a resposta em formato JSON com a classe prevista e a confiança.

---

## Observações Finais

Este projeto fornece uma solução completa para classificação de lesões dermatológicas em bovinos, desde o treinamento do modelo até sua utilização por meio de uma API Flask para predições. 

Com as explicações detalhadas fornecidas, você deve ser capaz de rodar e modificar o projeto conforme necessário.

Se tiver dúvidas ou encontrar algum problema, não hesite em entrar em contato.

--- 