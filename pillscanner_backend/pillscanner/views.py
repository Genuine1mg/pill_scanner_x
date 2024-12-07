from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import spacy

# Load the fine-tuned SpaCy model
nlp = spacy.load("./pillscanner/trainer/fine_tuned_med7_model")


class ExtractMedicineName(APIView):
    def post(self, request):
        text = request.data.get('text', '').strip()
        if not text:
            return Response({'error': 'No text provided or text is empty'}, status=status.HTTP_400_BAD_REQUEST)

        # Process the text with the model
        doc = nlp(text)

        # Print out all entities for debugging
        all_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        print(all_entities)  # Debugging step to see what entities are recognized

        # Extract MEDICINE entities
        medicines = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ == "MEDICINE"]

        # If medicines are found, return the best one based on length or confidence (you can tweak this logic)
        if medicines:
            # Sort by the length of the medicine name, or by confidence if available
            best_medicine = sorted(medicines, key=lambda x: len(x["text"]), reverse=True)[0]["text"]
            return Response({'medicine': best_medicine, 'entities': all_entities})
        else:
            return Response({'medicine': '', 'error': 'No medicine name found', 'entities': all_entities},
                            status=status.HTTP_200_OK)
