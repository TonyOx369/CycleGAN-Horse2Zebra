from django.shortcuts import render, redirect
from .forms import TranslationForm

def main_page(request):

    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            translation_option = form.cleaned_data['translation_option']
            has_verification_source = form.cleaned_data['verification_image'] is not None
            
            return redirect('translation', 
                            option=translation_option,
                            has_verification_source=has_verification_source)
    else:
        form = TranslationForm()
        
    return render(request, 'main_page.html', {'form':form})

def translation_page(request, option, has_verification):
    if request.method == 'POST':
        form = TranslationForm(request.POST, request.FILES)
        if form.is_valid():
            translated_image = form.save()
            # Process the translated image and verification image if available
            # Display the output image
            return render(request, 'translation_result.html', {'translated_image': translated_image})
    else:
        form = TranslationForm()
    return render(request, 'translation_page.html', {'form': form, 'has_verification': has_verification})