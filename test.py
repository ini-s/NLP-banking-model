import pickle
vectorizer, model = pickle.load(open("bank_model.pkl", "rb"))

text = input("Enter a text: ")
text_to_transform = [text]

X = vectorizer.transform(text_to_transform)
print(model.predict(X))
