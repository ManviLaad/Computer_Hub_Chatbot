import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Load intents from JSON file
intents_file = r'C:\Users\IT\Desktop\Computer_hub\new_intents.json'
with open(intents_file, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize lemmatizer and lists
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Process intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(classes))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open(r'C:\Users\IT\Desktop\Computer_hub\words.pkl', 'wb'))
pickle.dump(classes, open(r'C:\Users\IT\Desktop\Computer_hub\classes.pkl', 'wb'))

# Create training data
training = []
outputEmpty = [0] * len(classes)
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))


# Compile and train the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_simplilearnmodel.h5', hist)

print("Executed")
