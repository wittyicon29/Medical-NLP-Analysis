from transformers import BertTokenizer, BertForSequenceClassification
import torch

def analyze_sentiment_intent(text: str) -> dict:
    """
    Analyzes the sentiment and intent of the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing sentiment and intent predictions
    """
    # Load Sentiment Model
    sentiment_model = BertForSequenceClassification.from_pretrained("sentiment_model")
    sentiment_tokenizer = BertTokenizer.from_pretrained("sentiment_model")

    # Load Intent Model
    intent_model = BertForSequenceClassification.from_pretrained("intent_model")
    intent_tokenizer = BertTokenizer.from_pretrained("intent_model")

    # Define labels
    sentiment_labels = {0: "Anxious", 1: "Neutral", 2: "Reassured"}
    intent_labels = {0: "Seeking reassurance", 1: "Reporting symptoms", 2: "Expressing concern"}

    # Tokenize input
    inputs_sentiment = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs_intent = intent_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get predictions
    with torch.no_grad():
        sentiment_logits = sentiment_model(**inputs_sentiment).logits
        intent_logits = intent_model(**inputs_intent).logits

    # Get predicted label
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    intent_pred = torch.argmax(intent_logits, dim=1).item()

    # Return predictions as dictionary
    return {
        "Sentiment": sentiment_labels[sentiment_pred],
        "Intent": intent_labels[intent_pred]
    }

# Example usage:
if __name__ == "__main__":
    text = "I'm worried about my symptoms. Is this something serious?"
    result = analyze_sentiment_intent(text)
    print(result)