# from flask import Flask, request, jsonify
# import torch
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PIL import Image
# import joblib

# app = Flask(_name_)

# # Load the model and label encoder
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# resnet_model = models.resnet18(pretrained=False)
# num_features = resnet_model.fc.in_features
# resnet_model.fc = torch.nn.Linear(num_features, 3)  # Assuming 3 classes
# resnet_model.load_state_dict(torch.load('resnet_model.pth'))
# resnet_model.to(device)
# resnet_model.eval()

# label_encoder = joblib.load('label_encoder.pkl')

# # Define the transformation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def predict_stage(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = np.array(image) / 255.0
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = resnet_model(image)
#         predicted_probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
#         predicted_label = label_encoder.inverse_transform([torch.argmax(outputs, 1).item()])[0]
    
#     return predicted_label, predicted_probabilities

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file:
#         # Read the image file
#         image = Image.open(file.stream)
#         image = np.array(image)

#         # Predict the stage
#         predicted_label, predicted_probabilities = predict_stage(image)
        
#         # Return the result
#         return jsonify({'label': predicted_label, 'probabilities': predicted_probabilities.tolist()})

# if _name_ == '_main_':
#     app.run(debug=True)