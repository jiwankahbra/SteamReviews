import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import steamreviews
import gradio as gr
import re
import langid
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


app_id = 1172470

# Define the file path where you saved the filtered reviews
input_file_path = f'data/filtered_reviews2_{app_id}.json'

# Load the JSON file
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    loaded_filtered_reviews2 = json.load(json_file)

# Convert the list of dictionaries back to a list of tuples
filtered_reviews2 = [(item['review'], item['sentiment']) for item in loaded_filtered_reviews2]

print(f"Loaded {len(filtered_reviews2)} filtered reviews from {input_file_path}")

# i want to randomly select 50,000 reviews, 25,000 from postive sentiment and 25,000 from negative
postive_reviews = [review for review, sentiment in filtered_reviews2 if sentiment]
negative_reviews = [review for review, sentiment in filtered_reviews2 if not sentiment]

import random
random_positive_reviews = random.sample(postive_reviews, 25000)
random_negative_reviews = random.sample(negative_reviews, 25000)
Final_reviews = random_positive_reviews + random_negative_reviews

# Pair the reviews and sentiments
paired_final_reviews = [(review, 1) for review in random_positive_reviews] + [(review, 0) for review in random_negative_reviews]
# Shuffle the paired reviews and sentiments
random.shuffle(paired_final_reviews)

# Split back into reviews  and sentiments
X = [review for review, sentiment in paired_final_reviews]
Y = [sentiment for review, sentiment in paired_final_reviews]

#  Vectorize the text data using CountVectorizer
max_features = 10000  # Maximum number of features
vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
X_vectorized = vectorizer.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, random_state=42)

nn_model_9 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_9.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_9.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

#  Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_9.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 9 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

# Example query review
query_review = ["The game is good"]
query_vectorized = vectorizer.transform(query_review).toarray()

# Predict
prediction = nn_model_9.predict(query_vectorized)
sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
print(f"The sentiment is: {sentiment} and the percentage is {prediction[0]}")





# Define the function to fetch, clean, and preprocess reviews
def fetch_and_preprocess_reviews(app_id):
    print(f"Fetching reviews for app_id: {app_id}")
    try:
        # Fetch reviews using steamreviews (
        review_dict, _ = steamreviews.download_reviews_for_app_id(app_id, chosen_request_params={'language': 'english'})
        reviews = [review['review'] for review in review_dict['reviews'].values()]

        # Clean and preprocess the reviews
        # Remove empty reviews
        reviews = [review for review in reviews if review.strip()]

        # Remove duplicate reviews
        reviews = list(set(reviews))

        # Remove non-textual reviews
        filtered_reviews = []
        for review in reviews:
            # Remove reviews with too many non-alphanumeric characters
            if len(re.findall(r'\W', review)) / len(review) > 0.5:
                continue

            # Remove reviews with too many repeated characters
            if any(len(match.group(0)) > 3 for match in re.finditer(r'(.)\1+', review)):
                continue

            # Remove reviews that are too short
            if len(review.split()) < 3:
                continue

            filtered_reviews.append(review)

        # Remove reviews that are not in English using langid
        filtered_reviews = [review for review in filtered_reviews if langid.classify(review)[0] == 'en']

        print(f"Number of reviews after preprocessing: {len(filtered_reviews)}")
        return filtered_reviews
    except Exception as e:
        print(f"Error during fetching and preprocessing reviews: {e}")
        return []


# Define the function to create a wordcloud
def create_wordcloud(reviews):
    try:
        print("Creating word cloud...")
        # Add custom stopword 'game'
        custom_stop_words = ['game']

        # Use default English stopwords and combine with custom stopwords
        vectorizer2 = CountVectorizer(stop_words='english')
        all_stop_words = list(vectorizer2.get_stop_words()) + custom_stop_words

        # Vectorize the reviews
        vectorizer2 = CountVectorizer(stop_words=all_stop_words)
        X = vectorizer2.fit_transform(reviews)

        # Generate a Word Cloud from the term-document matrix
        tdm = X.toarray()
        word_freq = tdm.sum(axis=0)
        wordcloud = WordCloud(width=800, height=400, background_color='black',
                              colormap='Set2').generate_from_frequencies(
            dict(zip(vectorizer2.get_feature_names_out(), word_freq)))

        # Save the wordcloud to a file and return the path
        wordcloud_path = "wordcloud.png"
        wordcloud.to_file(wordcloud_path)
        print(f"Word cloud saved to {wordcloud_path}")
        return wordcloud_path
    except Exception as e:
        print(f"Error during word cloud generation: {e}")
        return None


# Use Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
        # Steam Review Sentiment Analysis
        Provide an app_id to fetch and preprocess the reviews, generate a word cloud, and predict sentiment for a given review.
    """)

    # Input for app_id
    app_id_input = gr.Number(label="Enter Steam app_id")
    fetch_reviews_button = gr.Button("Fetch and Preprocess Reviews")
    wordcloud_output = gr.Image(label="Word Cloud of Reviews")


    # Function to handle review fetching and preprocessing
    def handle_fetch_reviews(app_id):
        reviews = fetch_and_preprocess_reviews(app_id)
        if len(reviews) == 0:
            return None
        wordcloud_path = create_wordcloud(reviews)
        if wordcloud_path is None:
            return "Error: Unable to generate word cloud. Please try again."
        return wordcloud_path


    fetch_reviews_button.click(handle_fetch_reviews, inputs=app_id_input, outputs=wordcloud_output)

    # Input for sentiment prediction
    review_input = gr.Textbox(label="Enter a Review for Sentiment Prediction")
    sentiment_output = gr.Textbox(label="Predicted Sentiment")
    predict_button = gr.Button("Predict Sentiment")


    # Predict sentiment directly within the Gradio interface
    def handle_sentiment_prediction(review):
        try:
            print("Predicting sentiment...")
            # Use the already fitted vectorizer to transform the input review
            query_vectorized = vectorizer.transform([review]).toarray()

            # Predict sentiment
            prediction = nn_model_9.predict(query_vectorized)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            confidence = prediction if sentiment == "Positive" else 1 - prediction
            print(f"The sentiment is: {sentiment} and the confidence score is {confidence}")
            return f"{sentiment} (Confidence: {confidence:.2f})"
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error: Unable to predict sentiment. Please try again."


    predict_button.click(handle_sentiment_prediction, inputs=review_input, outputs=sentiment_output)

demo.launch()