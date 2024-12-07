import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from spacy.tokens import DocBin
import random

# Load the pre-trained Med7 model
nlp = spacy.load("en_core_med7_lg")

# Add the "MEDICINE" label to the NER pipeline if it's not already present
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add the custom entity label
ner.add_label("MEDICINE")

# Load your training data from 'train_data.spacy'
doc_bin = DocBin().from_disk("train_data.spacy")
train_data = []
for doc in doc_bin.get_docs(nlp.vocab):
    train_data.append((doc.text, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))

# Initialize the optimizer
optimizer = nlp.resume_training()

# Training loop
for epoch in range(10):  # Train for 10 epochs
    random.shuffle(train_data)
    losses = {}

    # Create mini-batches of the data
    batches = minibatch(train_data, size=16)
    for batch in batches:
        for text, annotations in batch:
            # Create an Example object and update the model
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.6, losses=losses)

    print(f"Epoch {epoch + 1}: Losses {losses}")

# Save the fine-tuned model
nlp.to_disk("fine_tuned_med7_model")
print("Fine-tuned Med7 model saved to 'fine_tuned_med7_model'")
