import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('IMDB Dataset.csv')
# Encode labels
from sklearn.preprocessing import LabelEncoder
df['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])

# Features & labels
X, y = df['review'].values, df['sentiment'].values

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Parameters
VOCAB_SIZE, MAX_LEN = 10000, 200

from tensorflow.keras.preprocessing.text import Tokenizer
# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

#converts words to corresponding sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(X_test_pad, y_test)
print('Test Accuracy:', accuracy)
print('Test Loss:', loss)

# Predictions
y_pred = (model.predict(X_test_pad) > 0.5).astype(int)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Report
print('Classification Report:\n',
      classification_report(y_test, y_pred, target_names=['negative', 'positive']))
