import json
from spacy.tokens import DocBin
import spacy

# Load your data
with open('medicine_data.json') as f:
    data = json.load(f)

# Load a SpaCy blank model for tokenization
nlp = spacy.blank("en")

# Convert to SpaCy's DocBin format
doc_bin = DocBin()

for item in data:
    text = item['text']
    entities = item['entities']
    doc = nlp.make_doc(text)
    ents = []

    # Manually adjust the entity spans to consider spaces
    for ent in entities:
        start, end, label = ent['start'], ent['end'], ent['label']

        # Adjust start and end indices to match SpaCy tokenization
        # Create a list of tokens including spaces
        tokenized_text = [token.text for token in doc]

        # Check if the span is correctly formed including spaces
        # Check character spans within the text, including spaces
        span = doc.char_span(start, end, label=label)

        if span is None:
            print(f"Skipping entity ({start}, {end}, {label}) in text: '{text}'")
            print(f"Tokens: {tokenized_text}")
            print(f"Start and end indices: {start} to {end}")
        else:
            ents.append(span)

    doc.ents = ents  # Set entities for the document
    doc_bin.add(doc)

# Save the DocBin to a file
doc_bin.to_disk("train_data.spacy")
print("Training data saved to 'train_data.spacy'")
