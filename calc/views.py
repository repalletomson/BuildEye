

import torch
import joblib
import numpy as np
from PIL import Image
from django.conf import settings
from torchvision import transforms
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

from django.views.decorators.csrf import csrf_exempt

from PIL import Image


from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .forms import ImageComparisonForm
import cv2
import numpy as np


@csrf_exempt
def image_comparsion_view(request):
    form=ImageComparisonForm()

    if request.method == 'POST':
        form = ImageComparisonForm(request.POST, request.FILES)
        if form.is_valid():
            initial_image = form.cleaned_data['initial_image']
            current_image = form.cleaned_data['current_image']
       # This variable is defined, but not used in this example

            fs = FileSystemStorage(location='uploads/')

            # Save initial image
            initial_filename = fs.save(initial_image.name, initial_image)
            initial_image_path = fs.path(initial_filename)

            # Save current image
            current_filename = fs.save(current_image.name, current_image)
            current_image_path = fs.path(current_filename)

            try:
                # Load images using OpenCV directly from the saved file paths
                img1 = cv2.imread(initial_image_path)
                img2 = cv2.imread(current_image_path)

                if img1 is None or img2 is None:
                    return JsonResponse({"error": "Error in loading images."}, status=400)

                # Resize and convert to grayscale
                height, width = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                resized_img1 = cv2.resize(img1, (width, height))
                resized_img2 = cv2.resize(img2, (width, height))
                gray_img1 = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2GRAY)
                gray_img2 = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2GRAY)

                # Compute difference and apply threshold
                diff = cv2.absdiff(gray_img1, gray_img2)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

                # Calculate percentage of changes
                total_pixels = np.prod(thresh.shape)
                changed_pixels = cv2.countNonZero(thresh)
                percentage_completed = (changed_pixels / total_pixels) * 100

                # Return the percentage as a JSON response
                return JsonResponse({"percentage_completed": percentage_completed})

            except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)
    else:
        form=ImageComparisonForm()
        return render(request, 'image_comparison.html', {'form': form})
def dashboard(request):
    return render(request,'dashboard.html')
def request_demo(request):  
    if request.method == "POST":
        # Process the form data
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        company = request.POST.get('company')
        email = request.POST.get('email')
        
        return HttpResponse(f"""
            <html>
                <body>
                    <h1>Thank you for your request, {first_name} {last_name}!</h1>
                    <a href="/" class="bg-yellow-500 text-white font-semibold py-2 px-4 rounded hover:bg-yellow-600 transition">Go to Home</a>
                </body>
            </html>
        """)
    return render(request, 'requestDemo.html')
def home(request):
    return render(request, 'base.html')
# def upload(request):
#     return render(request, 'check_progress.html')
# Load the model, label encoder, and GPT-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update the file paths to the absolute path
# model_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\resnet_model.pth'
# label_encoder_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\label_encoder.pkl'
model_path = os.path.join(settings.BASE_DIR, 'calc', 'data', 'resnet_model.pth')
label_encoder_path = os.path.join(settings.BASE_DIR, 'calc', 'data', 'label_encoder.pkl')

resnet_model = models.resnet18(pretrained=False)
num_features = resnet_model.fc.in_features
label_encoder = joblib.load(label_encoder_path)
resnet_model.fc = torch.nn.Linear(num_features, len(label_encoder.classes_))
resnet_model.load_state_dict(torch.load(model_path))
resnet_model.to(device)
resnet_model.eval()


# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the transformations for data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_text(prompt, num_sequences=1, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    outputs = gpt2_model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=num_sequences, 
        do_sample=True, 
        top_p=0.95, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def predict_stage(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = resnet_model(image)
        predicted_probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probabilities)])[0]

    return predicted_label

