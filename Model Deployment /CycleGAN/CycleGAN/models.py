from django.db import models

class TranslatedImage(models.Model):
    input_image = models.ImageField(upload_to='input_images/')
    translated_image = models.ImageField(upload_to='translated_images/')
    verification_image = models.ImageField(upload_to='verification_images/', null=True, blank=True)
    translation_option = models.CharField(max_length=50)

