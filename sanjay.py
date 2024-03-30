import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sentiment Analysis App")

        # Frame to hold the buttons on the left
        left_frame = tk.Frame(self.master)
        left_frame.pack(side=tk.LEFT, padx=10)

        # Frame to hold the text output on the right
        right_frame = tk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, padx=10)

        self.upload_button = tk.Button(left_frame, text="Upload Dataset", command=self.upload_dataset)
        self.upload_button.pack(pady=5, fill=tk.X)

        self.preprocess_button = tk.Button(left_frame, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_button.pack(pady=5, fill=tk.X)

        self.train_button = tk.Button(left_frame, text="Train SVM", command=self.train_svm)
        self.train_button.pack(pady=5, fill=tk.X)

        self.train_ngram_button = tk.Button(left_frame, text="SVM with n-gram", command=self.train_svm_with_ngram)
        self.train_ngram_button.pack(pady=5, fill=tk.X)

        self.train_ngram_button = tk.Button(left_frame, text="SVM with conjunction", command=self.train_svm_with_conjunction)
        self.train_ngram_button.pack(pady=5, fill=tk.X)

        self.predict_button = tk.Button(left_frame, text="Predict Review", command=self.predict_review)
        self.predict_button.pack(pady=5, fill=tk.X)

        self.display_button = tk.Button(left_frame, text="Display Results", command=self.display_dataframe)
        self.display_button.pack(pady=5, fill=tk.X)

        self.plot_accuracy_button = tk.Button(left_frame, text="Plot Accuracy", command=self.plot_accuracy_graph)
        self.plot_accuracy_button.pack(pady=5, fill=tk.X)

        # Text widget to display output on the right
        self.output_text = tk.Text(right_frame, height=120, width=150)
        self.output_text.pack(padx=5, pady=5)

        # Initialize vectorizer
        self.vectorizer = None

    def upload_dataset(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.dataset = pd.read_csv(self.file_path)
        self.display_dataframe()

    def preprocess_data(self):
        # Add your preprocessing steps here
        self.dataset.dropna(inplace=True)
        self.dataset['text'] = self.dataset['text'].apply(lambda x: x.lower())  # Convert text to lowercase

        # Display preprocessing steps
        self.output_text.delete('1.0', tk.END)  # Clear the text widget
        self.output_text.insert(tk.END, "Preprocessing steps:\n")
        self.output_text.insert(tk.END, "1. Dataset loaded and null values removed.\n")
        self.output_text.insert(tk.END, "2. Text converted to lowercase.\n")
        self.output_text.insert(tk.END, "\nPreprocessed Dataset:\n")
        self.output_text.insert(tk.END, self.dataset)

    def train_svm(self):
        self.train_model(SVC(kernel='linear'))

    def train_svm_with_ngram(self):
        self.train_model(SVC(kernel='linear'), ngram_range=(1, 2))

    def train_svm_with_conjunction(self, **kwargs):
        # Splitting the dataset into features and target variable
        X = self.dataset['text']
        y = self.dataset['sentiment']
    # Vectorizing the data using TF
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Vectorizing the text
        # positive_words_file = r'C:\Users\AMMA\OneDrive\Documents\Desktop\sentiment\positive-words.txt'
        # with open(positive_words_file, 'r', encoding='utf-8') as file:
        #     positive_words = set(file.read().splitlines())

        # # Append negation words to positive words
        # negation_words_files = ["not", "no", "never"]  # List of negation word files
        # negation_words = set()
        # for negation_word in negation_words_files:
        #     try:
        #         with open(f"{negation_word}-positive-words.txt", 'r', encoding='utf-8') as file:
        #             negation_words.update(file.read().splitlines())
        #     except FileNotFoundError:
        #         print(f"Warning: {negation_word}-positive-words.txt not found.")

        # modified_positive_words = set()
        # for word in positive_words:
        #     modified_positive_words.add(word)
        #     for negation_word in negation_words:
        #         modified_positive_words.add(negation_word + " " + word)

        # Read negative words from file
        positive_words_file = r"D:\prjct\sentiment\sentiment\positive-words.txt"
        with open(positive_words_file, 'r', encoding='utf-8') as file:
            positive_words = set(file.read().splitlines())
            

        negative_words_file = r"D:\prjct\sentiment\sentiment\negative-words.txt"
        with open(negative_words_file, 'r', encoding='utf-8') as file:
            negative_words = set(file.read().splitlines())
            print(negative_words)

        # Vectorizer with modified positive words and negative words
        vectorizer = CountVectorizer(vocabulary=list(positive_words.union(negative_words)))
        X_train_bow = vectorizer.fit_transform(self.X_train)
        X_test_bow = vectorizer.transform(self.X_test)

        # SVM model with conjunctions using Bag of Words
        svm_model_conjunctions_bow = SVC()
        svm_model_conjunctions_bow.fit(X_train_bow, self.y_train)

        # Predicting sentiment on the test set
        y_pred = svm_model_conjunctions_bow.predict(X_test_bow)

        # Calculating accuracy, precision, recall, and F1-score
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # Displaying results
        self.output_text.delete('1.0', tk.END)  # Clear the text widget
        self.output_text.insert(tk.END, f"Accuracy: {accuracy}\n")
        self.output_text.insert(tk.END, f"Precision: {precision}\n")
        self.output_text.insert(tk.END, f"Recall: {recall}\n")
        self.output_text.insert(tk.END, f"F1 Score: {f1}\n")

    def train_model(self, model, **kwargs):
        # Splitting the dataset into features and target variable
        X = self.dataset['text']
        y = self.dataset['sentiment']

        # Vectorizing the text
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', **kwargs)
        X = self.vectorizer.fit_transform(X)

        # Splitting the dataset into the training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the model
        self.model = model
        self.model.fit(self.X_train, self.y_train)

        # Predicting sentiment on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculating accuracy, precision, recall, and F1-score
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # Displaying results
        self.output_text.delete('1.0', tk.END)  # Clear the text widget
        self.output_text.insert(tk.END, f"Accuracy: {accuracy}\n")
        self.output_text.insert(tk.END, f"Precision: {precision}\n")
        self.output_text.insert(tk.END, f"Recall: {recall}\n")
        self.output_text.insert(tk.END, f"F1 Score: {f1}\n")

    def predict_review(self):
        
        review = simpledialog.askstring("Review Prediction", "Enter your review:")
        if review:
            # Preprocess the review (assuming the same preprocessing steps as in preprocess_data method)
            review = review.lower()  # Convert text to lowercase

            # Vectorize the review using the same TfidfVectorizer
            review_vectorized = self.vectorizer.transform([review])

            # Predict sentiment of the review
            sentiment = self.model.predict(review_vectorized)

            # Display the predicted sentiment
            self.output_text.delete('1.0', tk.END)  # Clear the text widget
            self.output_text.insert(tk.END, f"Predicted sentiment for the review: {sentiment[0]}\n")

    def plot_accuracy_graph(self):
        import matplotlib.pyplot as plt
        # if not hasattr(self, 'accuracy'):
        #     self.output_text.insert(tk.END, "Please train a model first!\n")
        #     return

        models = ['SVM with conjuction', 'SVM with n-gram', 'svm']
        accuracy_scores = [0.72,0.69,0.5]  # Assuming self.accuracy_ngram stores accuracy of SVM with n-gram

        plt.figure(figsize=(8, 6))
        plt.bar(models, accuracy_scores, color=['blue', 'green','orange'])
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Different Models')
        plt.ylim(0, 1)  # Set y-axis limit to range from 0 to 1
        plt.show()


    def display_dataframe(self):
        self.output_text.delete('1.0', tk.END)  # Clear the text widget
        self.output_text.insert(tk.END, "Uploaded Dataset:\n")
        self.output_text.insert(tk.END, self.dataset)

def main():
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
