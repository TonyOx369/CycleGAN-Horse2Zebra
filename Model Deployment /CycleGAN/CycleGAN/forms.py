from django import forms
from .models import TranslatedImage

class TranslationForm(forms.ModelForm):
    class Meta:
        model = TranslatedImage
        feilds = ['input_image', 
                  'verification_image',
                  'translation_option']
        widgets = {
            'translation_option': forms.RadioSelect
        }
        