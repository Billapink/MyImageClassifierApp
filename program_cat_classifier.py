import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Define the height and width of the image being processed
target_width = 200
target_height = 200

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained PyTorch model
model = CNNModel()
model.load_state_dict(torch.load('cat_classifier_model.pth'))
model.eval()

# model = torch.load('cat_classifier_model.pth')
# model.eval()

# Define the image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((target_width, target_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension
    return img

# Class labels
class_labels = ['non-cat', 'cat']

# Take a new image as input
new_image_path = ("puppy_image.jpg")

# Preprocess the image and make a prediction
img_tensor = preprocess_image(new_image_path)
with torch.no_grad():
    outputs = model(img_tensor)
    predicted_class = torch.argmax(outputs).item()

# Print the prediction
print("Prediction:", class_labels[predicted_class])
