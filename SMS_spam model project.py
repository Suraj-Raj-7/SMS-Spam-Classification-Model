

#Step 1: Importing Libraries
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download stopwords from NLTK
nltk.download('stopwords')


#Step 2: Load and Explore the Dataset
# Load the dataset
df = pd.read_csv('C:/Users/91906/Downloads/spam.csv', encoding='latin-1')

# Preview the dataset
df.head()

# Keep only the necessary columns
df = df[['v1', 'v2']]  # 'v1' is the label (spam/ham), 'v2' is the message

# Rename the columns for clarity
df.columns = ['label', 'message']

#Step 3: Data Preprocessing
# Text Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # Join the words back into a single string
    return ' '.join(words)

# Apply the preprocessing to all messages
df['message'] = df['message'].apply(preprocess_text)

# Check the first few processed messages
df.head()

#Step 4: Convert Text to Numerical Data
# Split the dataset into features (X) and labels (y)
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Step 5: Build and Train the Model
# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print the classification report
print(classification_report(y_test, y_pred))

#Step 6: Test with a New SMS
# Function to predict if a message is spam or ham
def predict_message(message):
    # Preprocess the message
    message = preprocess_text(message)

    # Convert to TF-IDF format
    message_tfidf = vectorizer.transform([message])

    # Predict the label (spam/ham)
    prediction = model.predict(message_tfidf)[0]

    # Return the prediction
    return 'Spam' if prediction == 1 else 'Ham'


# Test the model with a sample message
test_message1 = "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"
test_message2 =  "Oh k...i'm watching here:)"
print('The result is: ' + predict_message(test_message1))



