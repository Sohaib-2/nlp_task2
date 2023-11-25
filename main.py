import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def preprocess_text(text):
    encoded_text = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    return encoded_text


def predict_label(text):
    encoded_text = preprocess_text(text)
    output = model(**encoded_text)
    logits = output[0]
    predictions = logits.argmax(-1)
    probs = logits.softmax(-1)[:, 1:].squeeze()
    return predictions.item(), probs.item()


def explain_label(label):
    # Define label explanations
    label_explanations = {
        1: "The text is likely to be positive.",
        0: "The text is likely to be negative.",
    }

    # Return the explanation for the predicted label
    return label_explanations.get(label, "Unknown label")


# Create the interactive interface
st.title("Text Classification with Hugging Face")
st.markdown("---")

# Input text field with placeholder text
user_input = st.text_input("Enter text for classification:", placeholder="Enter your text here...")

# Submit button
if st.button("Predict"):
    if user_input:
        # Predict the label and probability
        predicted_label, probability = predict_label(user_input)

        # Display the prediction results and label explanation
        st.markdown("---")
        st.markdown("**Predicted Label:** " + str(predicted_label))
        st.markdown("**Probability:** " + str(probability))
        st.markdown("---")
        st.markdown("**Label Explanation:** " + explain_label(predicted_label))
