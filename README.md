# SMS Spam Detection using TensorFlow

This project develops and evaluates various machine learning and deep learning models to classify SMS messages as "ham" (legitimate) or "spam". It serves as a comprehensive guide to building a robust spam detection system, comparing traditional algorithms with modern neural network architectures.

![Spam vs Ham](https://i.imgur.com/uTNaA1g.png)

## Table of Contents
- [Project Summary](#project-summary)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approaches](#modeling-approaches)
  - [Baseline: Multinomial Naive Bayes](#1-baseline-multinomial-naive-bayes)
  - [Model 1: Custom Text Vectorization and Embedding](#2-model-1-custom-text-vectorization-and-embedding)
  - [Model 2: Bidirectional LSTM](#3-model-2-bidirectional-lstm)
  - [Model 3: Transfer Learning with Universal Sentence Encoder (USE)](#4-model-3-transfer-learning-with-universal-sentence-encoder-use)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)

## Project Summary
This project tackles the problem of SMS spam detection by implementing and comparing four distinct models. The process starts with loading and preprocessing the "SMS Spam Collection" dataset. This involves cleaning the data, numerically encoding the labels, and preparing it for the models.

A **Multinomial Naive Bayes** classifier serves as a baseline. Following this, three deep learning models are built using TensorFlow and Keras: a **custom text vectorization and embedding model**, a **Bidirectional LSTM (Bi-LSTM)** model to capture sequence context, and a **transfer learning model** leveraging the powerful pre-trained **Universal Sentence Encoder (USE)**.

Each model is trained and then evaluated on a held-out test set using key performance metrics like accuracy, precision, recall, and F1-score. The analysis focuses on the F1-score due to the imbalanced nature of the dataset, ultimately identifying the most effective approach for this classification task.

## Dataset
The project utilizes the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.
- **Link:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description:** The dataset contains 5,572 SMS messages in English, tagged as either "ham" (legitimate) or "spam".

## Installation
To run this project, clone the repository and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sms-spam-detection.git](https://github.com/your-username/sms-spam-detection.git)
    cd sms-spam-detection
    ```

2.  **Install the dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    If a `requirements.txt` file is not available, you can install the packages manually:
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow scikit-learn tensorflow_hub
    ```

## Usage
The entire workflow, from data loading to model evaluation, is contained within a single Jupyter Notebook or Python script.

1.  Place the `spam.csv` dataset in the root directory of the project.
2.  Run the Jupyter Notebook `SMS_Spam_Detection.ipynb` or the Python script `main.py`.

The script will:
- Load and preprocess the data.
- Train the four different models.
- Evaluate the models and print a summary of their performance metrics.
- Display visualizations for data distribution and results.

## Modeling Approaches

### 1. Baseline: Multinomial Naive Bayes
A simple and efficient probabilistic classifier that performs well for text classification. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.

### 2. Model 1: Custom Text Vectorization and Embedding
A deep learning model with the following architecture:
- **Input Layer:** Takes raw text strings.
- **TextVectorization Layer:** Standardizes, tokenizes, and converts text into integer sequences.
- **Embedding Layer:** Learns a dense vector representation for each word.
- **GlobalAveragePooling1D Layer:** Reduces the sequence data to a single vector.
- **Dense Layers:** A hidden layer with a ReLU activation and a final output layer with a sigmoid activation for binary classification.

### 3. Model 2: Bidirectional LSTM
A more advanced recurrent neural network (RNN) designed to capture sequential information and context from both forward and backward directions in the text.
- **Architecture:** Includes two Bidirectional LSTM layers followed by Dense layers for classification. A Dropout layer is included to prevent overfitting.

### 4. Model 3: Transfer Learning with Universal Sentence Encoder (USE)
This model leverages Google's pre-trained Universal Sentence Encoder from TensorFlow Hub to generate high-dimensional sentence embeddings. This transfer learning approach utilizes knowledge from a model trained on a massive corpus of text data.
- **Architecture:**
  - **Input Layer:** Takes raw text strings.
  - **KerasLayer (USE):** Converts sentences into 512-dimension vectors. This layer is set to `trainable=False`.
  - **Dense Layers:** A Dropout layer for regularization, followed by a hidden layer and a final sigmoid output layer.

## Results and Analysis
The performance of all four models was evaluated on the test set. The results are summarized below:

| Model                        | Accuracy | Precision | Recall | F1-Score |
| :--------------------------- | :------: | :-------: | :----: | :------: |
| **MultinomialNB Model** |  0.9785  |  1.0000   | 0.8389 |  0.9124  |
| **Custom-Vec-Embedding Model**|  0.9749  |  0.9351   | 0.8889 |  0.9114  |
| **Bidirectional-LSTM Model** |  0.9785  |  0.9490   | 0.9028 |  0.9253  |
| **USE-Transfer learning Model**| **0.9839** | **0.9664** | **0.9444** | **0.9553** |

<br>

![Model Comparison Chart](https://i.imgur.com/k2m3V0h.png)

### Key Observations:
- All models perform exceptionally well, with accuracies exceeding 97%.
- The **USE-Transfer learning Model** achieves the highest scores across all metrics, particularly the **F1-score (0.9553)**.
- The high F1-score is crucial as it indicates a healthy balance between precision and recall, which is vital for an imbalanced dataset like this one. It means the model is excellent at both identifying spam correctly and not misclassifying legitimate messages as spam.

## Conclusion
This project successfully demonstrates that while traditional machine learning models provide a strong baseline, advanced deep learning techniques offer superior performance for SMS spam detection. The transfer learning approach using the Universal Sentence Encoder proved to be the most effective strategy, delivering the most robust and reliable classification. This highlights the power of leveraging pre-trained models for NLP tasks.
