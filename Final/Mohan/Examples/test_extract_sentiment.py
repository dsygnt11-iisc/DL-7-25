if __name__ == "__main__":
    import sys
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Define working_dir or set it to the appropriate path
    working_dir = "/Users/mohanpanakam/Documents/Finetuning/finetuning"

    def predict_sentiment(text, model, tokenizer, device):
        """
        Predict sentiment for a given text using the trained model
        Returns: Dictionary containing prediction results including logits
        """
        try:
            # Prepare model
            model.eval()
            model = model.to(device)

            # Tokenize input
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=1)[0].item()

            # Map prediction to sentiment
            sentiment_map = {0: "positive", 1: "neutral", 2: "negative"}
            confidence = probs[0][predicted_class].item()

            return {
                "text": text,
                "sentiment": sentiment_map[predicted_class],
                "confidence": f"{confidence:.4f}",
                "logits": logits[0].cpu().numpy().tolist(),
                "probabilities": {
                    "positive": f"{probs[0][0].item():.4f}",
                    "neutral": f"{probs[0][1].item():.4f}",
                    "negative": f"{probs[0][2].item():.4f}"
                }
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None


    # If arguments are provided, use them as input text
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        model_path = f"{working_dir}/goldbert"
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        result = predict_sentiment(input_text, model, tokenizer, device)
        if result:
            print(f"\nText: {result['text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Logits: {result['logits']}")
            print(f"Class Probabilities: {result['probabilities']}")
        else:
            print("Prediction failed.")
    else:
        print("Usage: python finbert_finetune_refactored.py <your text here>")