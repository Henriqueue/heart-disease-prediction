# heart-disease-prediction
# 🚀 Heart Disease Prediction Project
<img src="https://domf5oio6qrcr.cloudfront.net/medialibrary/6037/6c53ac6b-cfba-42fc-9f0b-6c96e1b6a637.jpg" alt="6c53ac6b-cfba-42fc-9f0b-6c96e1b6a637"/><img width="1200" height="699" alt="image" src="https://github.com/user-attachments/assets/69ec1ba3-2fcb-4e32-96d6-c4fb9c618138" />

This repository contains a classic machine learning project focused on predicting the presence of heart disease based on clinical attributes from the Cleveland dataset.

## 🎯 Objective

The primary goal of this project is to practice a fundamental end-to-end machine learning pipeline, including:
* Data preprocessing and normalization.
* Training multiple classic classification models.
* Evaluating and comparing model performance.
* Interpreting key metrics like the Confusion Matrix and Classification Report.

## 📊 Dataset

* **Source:** [Kaggle - Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)
* **File:** `heart.csv`
* **Target Variable:** `target` (0 = No Disease, 1 = Presence of Disease)

## 🛠️ Methodology

The analysis is documented in the `heart_disease_analysis.ipynb` notebook and follows these steps:

1.  **Data Loading:** The `heart.csv` dataset is loaded using Pandas.
2.  **Data Splitting:** The dataset is split into training (70%) and testing (30%) sets.
3.  **Preprocessing:** A `StandardScaler` is used to normalize the features. This is crucial for models like Logistic Regression and KNN.
4.  **Model Training:** Three different classifiers were trained on the scaled training data:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Decision Tree
5.  **Evaluation:** The models were evaluated on the test set using:
    * **Accuracy Score** for a high-level comparison.
    * **Confusion Matrix** to analyze Fals_ Positives vs. False Negatives.
    * **Classification Report** to assess Precision, Recall, and F1-Score for each class.
6.  **Pipeline Implementation:** The process was also encapsulated in a scikit-learn `Pipeline` to demonstrate a more robust and professional workflow that prevents data leakage.

## 📈 Results

The models yielded the following performance on the unseen test set:

| Model | Test Accuracy |
| :--- | :--- |
| Logistic Regression | [Sua Acurácia]% |
| K-Nearest Neighbors | [Sua Acurácia]% |
| Decision Tree | [Sua Acurácia]% |

*(Aqui, adicione uma breve conclusão. Exemplo:)*
**Conclusion:** The K-Nearest Neighbors model achieved the highest accuracy, making it the most effective classifier for this specific problem setup. The Logistic Regression also showed strong, balanced performance, while the Decision Tree was prone to overfitting.

## 🔧 How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[SeuUsuario]/heart-disease-prediction.git
    ```
2.  Navigate to the directory:
    ```bash
    cd heart-disease-prediction
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
4.  Open and run the Jupyter Notebook:
    ```bash
    jupyter notebook heart_disease_analysis.ipynb
    ```

    <h1>Tradução para Português:</h1>
# 🚀 Análise Preditiva de Doença Cardíaca (Cleveland)

Este repositório contém um projeto de Machine Learning para prever a presença de doença cardíaca com base em atributos clínicos, utilizando o dataset clássico de Cleveland.

## 🎯 Objetivo

O objetivo principal foi praticar um pipeline básico de ML, desde a preparação dos dados até a avaliação de diferentes modelos de classificação.

## 📊 Dataset

* **Fonte Original:** [Kaggle - Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)
* **Arquivo:** `heart.csv`
* **Variável Alvo:** `target` (0 = Sem doença, 1 = Presença de doença)

## 🛠️ Metodologia

O notebook `analise_heart_disease.ipynb` (ou o nome que você der) segue os seguintes passos:

1.  **Análise Exploratória (EDA):** Verificação inicial dos dados.
2.  **Pré-processamento:**
    * Separação da base em Treino (70%) e Teste (30%).
    * **Normalização** dos dados (StandardScaler) para otimizar os modelos.
3.  **Treinamento de Modelos:**
    * Regressão Logística
    * K-Nearest Neighbors (KNN)
    * Árvore de Decisão
4.  **Avaliação:**
    * Comparação de **Acurácia**.
    * Análise da **Matriz de Confusão** (Falsos Positivos vs. Falsos Negativos).
    * Análise do **Classification Report** (Precisão, Recall, F1-Score).

## 📈 Resultados (Exemplo)

| Modelo | Acurácia (Teste) |
| :--- | :--- |
| Regressão Logística | 85.71% |
| KNN (k=5) | 86.81% |
| Árvore de Decisão | 79.12% |

*(Adicione aqui suas conclusões, como "O modelo KNN teve o melhor desempenho..." ou "A Regressseão Logística teve um bom equilíbrio...")*

## 🔧 Como Executar

1.  Clone este repositório.
2.  Tenha o Python e as bibliotecas (pandas, scikit-learn) instalados.
3.  Abra o Jupyter Notebook ou rode o script.
