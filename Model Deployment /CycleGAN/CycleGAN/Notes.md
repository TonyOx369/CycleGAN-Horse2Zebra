1. Main Page:
   1. Form
      1. Horse to Zebra
      2. Zebra to Horse
   2. Do they have verification source

- Form Submission
  - The form data is sent to the server using a POST request.
  - "main_page" handles this request. It validates the form data
  - If form data is valid, move to "translation_page" with selected options and information about verification source

2. Translation Page:
   1. Form
      1. Upload an image for translation
      2. if verification source:
         1. upload the verification image

- Translation Processing:
  - The form data is sent to server using a POST request 
  - "translation_page" handles this request. 
  - saves the uploaded images and processed them using the model

3. Result Page:
   1. "translation_result" users can see the translated image. 
   2. If the verification source is given, display that too here