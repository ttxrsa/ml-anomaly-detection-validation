# ml-anomaly-detection-validation
Machine learning model validation for anomaly detection in induction motors, developed as part of a Master's research project.

## Dataset
https://engineering.case.edu/bearingdatacenter/download-data-file

## Validação de Modelos para Detecção de Anomalias

Este repositório contém experimentos preliminares para avaliação de modelos de detecção de anomalias em sinais de vibração de motores elétricos utilizando o dataset CWRU (Case Western Reserve University Bearing Data Center).

### Pipeline

O pipeline utilizado nos experimentos consiste em:

1. Carregamento dos sinais de vibração do Drive End (DE), lado onde a carga é acoplada;
2. Segmentação dos sinais em janelas de 2048 amostras;
3. Extração das características:
   - RMS;
   - Desvio padrão;
   - Curtose;
   - Skewness;
   - Pico a pico;
4. Normalização com StandardScaler;
5. Treinamento utilizando dados da condição normal;
6. Detecção de anomalias;
7. Avaliação por Precision, Recall, F1-score, Specificity e AUC-ROC.

### Modelos avaliados

Foram avaliados:

- Isolation Forest
- One-Class SVM
- PCA Reconstruction

Cada modelo foi inicialmente executado com uma configuração baseline e posteriormente submetido à otimização de hiperparâmetros.

### Resultados preliminares

| Modelo | F1 Baseline | F1 Otimizado | FP | FN |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.980311 | 0.996839 | 3 | 0 |
| One-Class SVM | 0.977273 | 0.993697 | 6 | 0 |
| PCA Reconstruction | 0.989540 | 0.998944 | 1 | 0 |

O PCA Reconstruction apresentou o maior F1-score no protocolo experimental inicial.


