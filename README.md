

# **SMS Spam Detection System**

An intelligent system leveraging Natural Language Processing (NLP) and Machine Learning to classify SMS messages as spam or ham (non-spam). This project demonstrates the application of text preprocessing, feature extraction, and predictive modeling to tackle the real-world issue of SMS spam.

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Design](#system-design)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**

With the increasing use of mobile messaging, spam SMS has become a significant problem, wasting time and potentially exposing users to scams and malicious activities. The SMS Spam Detection System is a solution that classifies SMS messages into two categories:
- **Spam**: Unwanted or malicious messages.
- **Ham**: Genuine messages.

By using machine learning models, this system offers an automated way to filter and block spam messages, enhancing the security and usability of messaging systems.

---

## **Features**

- **Data Preprocessing**: Cleans and prepares raw SMS data, including tokenization, stopword removal, stemming, and lemmatization.
- **Feature Extraction**: Converts text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
- **Machine Learning Models**: Implements models like Naïve Bayes and Support Vector Machines (SVM) for classification.
- **Evaluation Metrics**: Analyzes the performance of models using accuracy, precision, recall, and F1-score.
- **Spam Classification**: Predicts whether a new SMS message is spam or ham with high accuracy.

---

## **Technologies Used**

### **Programming Language**
- Python 3.x

### **Libraries and Tools**
- `scikit-learn`: For machine learning model training and evaluation.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `nltk`: For natural language processing tasks.
- `matplotlib` and `seaborn`: For data visualization.

### **Dataset**
- [SMS Spam Collection Dataset](https://www.kaggle.com/): A public dataset used for training and evaluation.

---

## **System Design**

The system follows a structured pipeline:
1. **Data Input**: Raw SMS messages are provided as input.
2. **Preprocessing**: Text data is cleaned and standardized.
3. **Feature Extraction**: Text is converted into numerical format using TF-IDF.
4. **Model Training**: Machine learning models (Naïve Bayes and SVM) are trained.
5. **Prediction**: Trained models classify new SMS messages as spam or ham.
6. **Evaluation**: The performance of the models is measured using evaluation metrics.

---

## **Installation**

Follow these steps to set up the project:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**
   - Download the SMS Spam Collection Dataset from Kaggle or any other source.
   - Place it in the `data/` directory.

---

## **Usage**

1. **Run the Preprocessing Script**
   ```bash
   python preprocess.py
   ```

2. **Train the Models**
   ```bash
   python train.py
   ```

3. **Test the Models**
   ```bash
   python evaluate.py
   ```

4. **Classify New Messages**
   Use the prediction script to classify new SMS messages:
   ```bash
   python predict.py "Your SMS message here"
   ```

---

## **Future Enhancements**

- **Deep Learning**: Implement advanced models like RNNs, LSTMs, and Transformers for better accuracy.
- **Real-Time Detection**: Optimize the system for real-time spam classification.
- **Multilingual Support**: Extend the system to handle SMS data in multiple languages.
- **Dataset Expansion**: Incorporate larger and more diverse datasets for better generalization.
- **API Deployment**: Package the system into an API for integration with messaging platforms.

---

## **Contributing**

Contributions are welcome! Follow these steps to contribute:
1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to the branch.
4. Open a pull request.

---

## **License**

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the license terms.

