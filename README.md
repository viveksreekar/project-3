# Project-3

------------------------------------------
#📊 1. Exploratory Data Analysis (EDA) on Zomato Bangalore Restaurants

This project performs a deep exploratory data analysis on the Zomato Bangalore Restaurants dataset to uncover key factors driving restaurant success. The analysis focuses on ratings, cost, location, and service offerings to provide actionable insights.

* **Key Features:**
    * Comprehensive data cleaning and preprocessing pipeline.
    * In-depth analysis of relationships between restaurant ratings, cost, location, and services (online ordering, table booking).
    * Rich visualizations using Matplotlib and Seaborn to illustrate trends and distributions.
    * Identifies key drivers for higher ratings, such as the availability of table booking and dining type.

* **Skills & Concepts Demonstrated:**
    * **Data Analysis:** Pandas, NumPy
    * **Data Visualization:** Matplotlib, Seaborn
    * **Techniques:** Data Cleaning, Data Wrangling, Statistical Analysis, Data Storytelling

* **How to Run:**
    1.  Clone the repository.
    2.  Install the required libraries: `pip install pandas matplotlib seaborn jupyter`
    3.  Download the dataset from the Kaggle link below and place it in the project directory.
    4.  Run the Jupyter Notebook or Python script to see the full analysis.

* **Dataset Link:** [Zomato Bangalore Restaurants on Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)

---

#🤖 2. Fake News Detection Model

This project builds a machine learning model to classify news articles as "REAL" or "FAKE". It uses Natural Language Processing (NLP) techniques to process the text and a linear classifier to make predictions, achieving high accuracy on the test set.

* **Key Features:**
    * An end-to-end NLP pipeline from raw text to classification.
    * Efficient text preprocessing using NLTK for stopword removal and stemming.
    * Uses TF-IDF (Term Frequency-Inverse Document Frequency) for robust feature extraction.
    * Employs a `PassiveAggressiveClassifier`, well-suited for text classification tasks.

* **Skills & Concepts Demonstrated:**
    * **Machine Learning:** Scikit-learn, Pandas
    * **NLP:** NLTK, Text Preprocessing, Feature Extraction (TF-IDF)
    * **Models:** Classification Models (PassiveAggressiveClassifier)

* **How to Run:**
    1.  Clone the repository.
    2.  Install the required libraries: `pip install pandas scikit-learn nltk`
    3.  Download the dataset and ensure `true.csv` and `fake.csv` are in the root directory.
    4.  Run the script: `python fake_news_detection_model.py`

* **Dataset Link:** [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

#📸 3. Image Caption Generator

This advanced deep learning project combines Computer Vision and NLP to generate descriptive captions for images. It uses a CNN-LSTM encoder-decoder architecture to understand the content of an image and translate that understanding into natural language.

* **Key Features:**
    * Implements a sophisticated **Encoder-Decoder** model.
    * Uses a pre-trained **VGG16 (CNN)** as the encoder for image feature extraction (Transfer Learning).
    * Employs an **LSTM (RNN)** as the decoder to generate sequential text captions.
    * The complete pipeline includes image preprocessing, text tokenization, model training, and inference.

* **Skills & Concepts Demonstrated:**
    * **Deep Learning:** TensorFlow, Keras
    * **Computer Vision (CV):** Convolutional Neural Networks (CNNs), Image Preprocessing
    * **NLP:** Recurrent Neural Networks (RNNs), LSTMs, Text Tokenization
    * **Architecture:** Encoder-Decoder Models, Transfer Learning

* **How to Run:**
    1.  Clone the repository.
    2.  Install the required libraries: `pip install tensorflow Pillow numpy tqdm`
    3.  Download the **Flickr8k** dataset and place the `Flicker8k_Dataset` folder and `Flickr8k.token.txt` file in the project directory.
    4.  **Note:** Training this model is computationally intensive and requires a GPU. It's highly recommended to run it on a platform like Google Colab.
    5.  Run the Python script to start the training process or generate captions using a pre-trained model.

* **Dataset Link:** [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
