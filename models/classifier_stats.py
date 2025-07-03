import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

# Load dataset and model
df = pd.read_csv("data/chatbot_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], test_size=0.2, random_state=42)
model = joblib.load("models/intent_classifier.joblib")

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=model.classes_))

# === 1. Plot class distribution ===
plt.figure(figsize=(6, 4))
df['intent'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Intent Class Distribution")
plt.xlabel("Intent")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === 2. Confusion matrix (absolute + normalized) ===
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.show()

# === 3. Show top features per class ===
vectorizer = model.named_steps['tfidfvectorizer']
classifier = model.named_steps['logisticregression']
feature_names = np.array(vectorizer.get_feature_names_out())

def show_top_features(class_index, class_name, top_n=10):
    top = np.argsort(classifier.coef_[class_index])[-top_n:]
    print(f"\nTop features for class '{class_name}':")
    for i in reversed(top):
        print(f"  {feature_names[i]:<20} {classifier.coef_[class_index][i]:.3f}")

for idx, class_name in enumerate(model.classes_):
    show_top_features(idx, class_name)

# === 4. t-SNE plot of test set vectors ===
X_test_vectors = vectorizer.transform(X_test)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_test_2d = tsne.fit_transform(X_test_vectors.toarray())

plt.figure(figsize=(8, 6))
for intent in np.unique(y_test):
    idxs = y_test == intent
    plt.scatter(X_test_2d[idxs, 0], X_test_2d[idxs, 1], label=intent, alpha=0.6)
plt.legend()
plt.title("t-SNE Visualization of Test Samples")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.show()

# === 5. Show misclassified examples ===
misclassified = X_test[y_test != y_pred]
for i, (text, true, pred) in enumerate(zip(misclassified, y_test[y_test != y_pred], y_pred[y_test != y_pred])):
    print(f"\nExample {i+1}")
    print(f"Text       : {text}")
    print(f"True label : {true}")
    print(f"Predicted  : {pred}")
    if i == 4:  # Show only first 5
        break
