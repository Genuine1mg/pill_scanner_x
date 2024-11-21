from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import spacy

nlp = spacy.load("en_core_med7_lg")


class ExtractMedicineName(APIView):
    def post(self, request):
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)

        doc = nlp(text)
        medicines = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ == "DRUG"]

        if medicines:
            return Response({'medicine': medicines[0]})
        else:
            return Response({'error': 'No medicine name found'}, status=status.HTTP_404_NOT_FOUND)
