Email/SMS Spam Classifier

A machine learning-powered web application to classify messages as Spam or Not Spam. Built using Streamlit for the interface, this app is designed to predict the nature of an email or SMS message based on its content.



Features

Classifies messages as Spam or Not Spam with a trained machine learning model.

Simple and intuitive user interface using Streamlit.

Supports both email and SMS classification.

Pre-trained model (pickle file) ensures quick and accurate predictions.





Getting Started

Follow these instructions to get the app up and running on your local machine.

1. Prerequisites

Ensure you have the following installed:

Python 3.8+

pip (Python package manager)

2. Install Dependencies

Use the provided requirements.txt to install all required Python packages.

pip install -r requirements.txt

3. Run the Application

Run the Streamlit application using the command below:

streamlit run app.py

4. Access the App

After running the app, it will be accessible on your local machine at:

http://localhost:8501



Usage

1. Enter the message you want to classify in the input text box.


2. Click the Predict button.


3. The app will display whether the message is Spam or Not Spam.




Technologies Used

Python

Streamlit

Scikit-learn

Pickle (for saving/loading models)




How It Works

1. The text input is preprocessed (if needed, using a cleaning function).


2. The processed text is transformed using a pre-trained TF-IDF Vectorizer (vectorizer.pkl).


3. The vectorized input is passed to the trained ML model (model.pkl) to predict whether the message is spam.



Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to:

Open an issue

Create a pull request



Acknowledgments

Special thanks to the developers of Streamlit, Scikit-learn, and the contributors to the open-source community.




