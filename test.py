import pickle
vectorizer, model = pickle.load(open("bank_model.pkl", "rb"))

while True:
    text = input("Enter a text (or click on 'q' to quite): ")
    text_to_transform = [text]

    if text.lower().strip() in ['q', 'quit', 'exit']:
        break

    X = vectorizer.transform(text_to_transform)
    pred = model.predict(X)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X).tolist()[0]
        print(
            f'prediction:{pred}, \nprobs:{[f"{prob:.2f}" for prob in probs]}')

    print(pred)
