# now for nn_models
import json
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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

# Split back into reviews (X) and sentiments (Y)
X = [review for review, sentiment in paired_final_reviews]
Y = [sentiment for review, sentiment in paired_final_reviews]

# Vectorize the text data using CountVectorizer
# Maximum number of features
max_features = 10000
vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
X_vectorized = vectorizer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split

#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, random_state=42)


# intial model
nn_model_1 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_1.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_1.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Step 5: Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_1.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 2 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_2 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_2.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_2.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_2.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 2 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_3 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_3.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_3.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_3.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 3 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_4 = Sequential([
    Dense(32, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(16, activation='tanh'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_4.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_4.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_4.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 4 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_5 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_5.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_5.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_5.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 5 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_6 = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(32, activation='tanh'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_6.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_6.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_6.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 6 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_7 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_7.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_7.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_7.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 7 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_8 = Sequential([
    Dense(128, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_8.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_8.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_8.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 8 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

nn_model_9 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#compile model
nn_model_9.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_9.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
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

nn_model_10 = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='tanh'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
])

#compile model
nn_model_10.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model_10.fit(X_train, np.array(Y_train), epochs=5, validation_data=(X_test, np.array(Y_test)), batch_size=64)

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Generate predictions and evaluate the model
Y_pred = (nn_model_10.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model 10 Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.show()

