import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
import gradio as gr

# Define the model architecture (same as in your notebook)
# Create the PyTorch sequential model
def create_model():
    # Load pre-trained VGG16 weights
    vgg = models.vgg16(weights='IMAGENET1K_V1')
    
    # Freeze VGG16 layers
    for param in vgg.parameters():
        param.requires_grad = False
    
    # Modify VGG16 to remove the classifier
    vgg.classifier = nn.Identity()
    
    return nn.Sequential(
        vgg,
        nn.Flatten(),
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(0.33),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.33),
        nn.Linear(128, 3),
        nn.Softmax(dim=1)
    )

# Initialize model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model().to(device)
model.load_state_dict(torch.load('mask_detection_model.pth', map_location=device))
model.eval()

# Load face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Categories for prediction
categories = ['Incorrect Mask', 'With Mask', 'Without Mask']

def predict(image):
    if image is None:
        return "No image provided", None
    
    # Convert from BGR to RGB if image is from OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Create a copy of the image for drawing
    output_image = image.copy()
    
    results = []
    
    # Process each detected face
    if len(faces) == 0:
        return "No faces detected", output_image

    for (x, y, w, h) in faces:
        # Extract and preprocess the face region
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        
        # Transform for model
        face_tensor = transform(face).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(face_tensor)
            pred_idx = torch.argmax(output).item()
            confidence = output[0][pred_idx].item() * 100
        
        # Get prediction label and color
        label = categories[pred_idx]
        if pred_idx == 1:  # With Mask
            color = (0, 255, 0)  # Green
        elif pred_idx == 0:  # Incorrect Mask
            color = (255, 165, 0)  # Orange
        else:  # Without Mask
            color = (255, 0, 0)  # Red
        
        # Draw rectangle and label on the image
        cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_image, f"{label} ({confidence:.1f}%)", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        results.append(f"{label}: {confidence:.1f}%")
    
    return "\n".join(results), output_image

# Custom CSS for neon theme
custom_css = """
/* Neon Theme Variables */
:root {
    --neon-green: #39FF14;
    --neon-green-glow: rgba(57, 255, 20, 0.6);
    --neon-green-dark: #32CD32;
    --background: #000000;
    --background-alt: #111111;
    --text-glow: 0 0 10px var(--neon-green-glow),
                 0 0 20px var(--neon-green-glow),
                 0 0 30px var(--neon-green-glow);
    --box-glow: 0 0 5px var(--neon-green-glow),
                0 0 10px var(--neon-green-glow),
                0 0 15px var(--neon-green-glow);
    --primary-color: var(--neon-green);
    --primary-light: var(--neon-green-glow);
    --primary-dark: var(--neon-green-dark);
    --secondary-color: var(--neon-green);
    --accent-color: var(--neon-green);
    --text-light: var(--neon-green);
    --text-dark: var(--background);
}

#header {
    text-align: center;
    margin-bottom: 15px;
    background: var(--background);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid var(--neon-green);
    box-shadow: var(--box-glow);
    animation: neon-pulse 2s infinite;
}

@keyframes neon-pulse {
    0%, 100% { box-shadow: var(--box-glow); }
    50% { box-shadow: 0 0 10px var(--neon-green-glow),
                      0 0 15px var(--neon-green-glow),
                      0 0 20px var(--neon-green-glow); }
}

#header h1 {
    color: var(--neon-green);
    font-size: 2.2em;
    margin: 0;
    font-weight: 800;
    text-shadow: var(--text-glow);
}

#header h3 {
    color: var(--neon-green);
    margin-top: 5px;
    margin-bottom: 0;
    font-size: 1.1em;
    text-shadow: 0 0 5px var(--neon-green-glow);
}

#input-image, #output-image {
    border-radius: 10px;
    border: 2px solid var(--neon-green) !important;
    padding: 10px;
    background: var(--background) !important;
    box-shadow: var(--box-glow);
}

#prediction-container {
    background: var(--background);
    border: 1px solid var(--neon-green);
    border-radius: 12px;
    padding: 20px;
    margin: 20px auto;
    max-width: 90%;
    box-shadow: var(--box-glow);
}

#prediction-box {
    font-size: 1.3em;
    text-align: center;
    color: var(--neon-green);
    font-weight: bold;
    text-shadow: var(--text-glow);
    background: transparent !important;
    border: none !important;
}

#prediction-box textarea {
    background: transparent !important;
    border: none !important;
    color: var(--neon-green) !important;
    text-align: center;
    text-shadow: 0 0 5px var(--neon-green-glow);
}

#controls {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

#detect-button {
    min-width: 200px;
    font-size: 1.2em;
    background: var(--background) !important;
    border: 1px solid var(--neon-green) !important;
    box-shadow: var(--box-glow);
    transition: all 0.3s ease;
    padding: 12px 24px;
    border-radius: 8px;
    color: var(--neon-green) !important;
    text-shadow: var(--text-glow);
}

#detect-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 15px var(--neon-green-glow),
                0 0 25px var(--neon-green-glow),
                0 0 35px var(--neon-green-glow);
}

#detect-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
}

#instructions-container {
    background: var(--background);
    border: 1px solid var(--neon-green);
    border-radius: 15px;
    padding: 25px;
    margin: 20px auto;
    box-shadow: var(--box-glow);
    color: var(--neon-green);
}

#instructions-container h3 {
    color: var(--neon-green);
    border-bottom: 1px solid var(--neon-green);
    padding-bottom: 8px;
    margin-bottom: 15px;
    text-shadow: var(--text-glow);
}

#instructions-container ul, #instructions-container p {
    color: var(--neon-green);
    line-height: 1.6;
    text-shadow: 0 0 3px var(--neon-green-glow);
}

/* Add neon pulse animations */
@keyframes neon-text-pulse {
    0%, 100% { text-shadow: var(--text-glow); }
    50% { text-shadow: 0 0 15px var(--neon-green-glow),
                       0 0 25px var(--neon-green-glow),
                       0 0 35px var(--neon-green-glow); }
}

.gradio-container {
    background: var(--background) !important;
}

/* Custom styling for emoji indicators */
#instructions-container li span {
    text-shadow: var(--text-glow);
    font-weight: bold;
}

/* Add hover effects to interactive elements */
#instructions-container:hover,
#prediction-container:hover,
#input-image:hover,
#output-image:hover {
    box-shadow: 0 0 15px var(--neon-green-glow),
                0 0 25px var(--neon-green-glow),
                0 0 35px var(--neon-green-glow);
    transition: box-shadow 0.3s ease;
}
"""

# Create Gradio interface with custom theme
demo = gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    ),
    css=custom_css
)

with demo:
    # Header with styled title
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # 😷 Face Mask Detection System
            ### Powered by Deep Learning & Computer Vision
            """
        )
    
    # Main content area
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            input_image = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="📸 Live Camera Feed",
                elem_id="input-image"
            )
        
        # Right column - Output
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🔍 Detection Results",
                elem_id="output-image"
            )
    
    # Control buttons
    with gr.Row(elem_id="controls"):
        trigger = gr.Button(
            "🎯 Detect Masks",
            size="lg",
            variant="primary",
            elem_id="detect-button"
        )
    
    # Prediction area with container
    with gr.Row(elem_id="prediction-container"):
        output_text = gr.Textbox(
            label="Prediction Results",
            elem_id="prediction-box",
            show_label=False,
            container=False
        )
    
    # Instructions and Information in a container
    with gr.Row(elem_id="instructions-container"):
        gr.Markdown(
            """
            ### 📋 Instructions:
            1. Click "**Enable Webcam**" to start your camera
            2. Position yourself or others in the camera view
            3. Click "**Detect Masks**" to analyze mask wearing
            4. View results in real-time with bounding boxes:
               - 🟢 Green: Properly wearing mask
               - 🟡 Orange: Incorrectly wearing mask
               - 🔴 Red: No mask detected

            ### ℹ️ Important Notes:
            - For best results, ensure good lighting and face the camera directly
            - Multiple faces can be detected simultaneously
            - Each detection includes confidence percentage
            
            ### 🔄 Real-time Detection Limitations:
            - Due to Hugging Face Spaces restrictions, continuous real-time detection is not available
            - Each detection requires a manual button click
            - For real-time detection, consider running this app locally
            
            ### 💡 Technical Details:
            - Model: VGG16-based CNN with transfer learning
            - Face Detection: Haar Cascade Classifier
            - Processing: Server-side inference with PyTorch
            """
        )

    trigger.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_text, output_image],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)