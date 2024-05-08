import streamlit as st
import tensorflow as tf
import numpy as np

# Load the saved model
model_path = "potatoes.h5"  # Assuming "potatoes.h5" is in the same directory as this script
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Define class information
class_info = {
    'Potato___Early_blight': {
        'emoji': '⚠️',
        'description': 'Early blight is a common foliage disease that affects potatoes, typically observed on older leaves.',
        'management': 'Ensure good air circulation around plants, avoid overhead watering, and apply fungicides as necessary.'
    },
    'Potato___Late_blight': {
        'emoji': '⚠️',
        'description': 'Late blight is a serious potato disease that can cause rapid and extensive damage to potato foliage and tubers.',
        'management': 'Apply fungicides preventively and practice crop rotation to reduce disease pressure.'
    },
    'Potato___healthy': {
        'emoji': '✅',
        'description': 'The potato plant appears healthy without any visible signs of disease.',
        'management': 'Maintain proper soil moisture, nutrition, and pest control measures to support plant health.'
    }
}

# Function to preprocess and predict image
def predict_image(image):
    # Preprocess image
    img_array = tf.image.decode_image(image, channels=3)
    img_array = tf.image.resize(img_array, [256, 256])
    img_array = tf.expand_dims(img_array, 0)  # Expand dimensions to create batch
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    
    return predicted_class, confidence

# Streamlit UI
st.title("Potato Disease Classification")

# Sidebar for input
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display result
if uploaded_file is not None:
    # Display uploaded image
    image = uploaded_file.read()
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict
    predicted_class, confidence = predict_image(image)
    
    # Display result
    st.write(f"Prediction: {predicted_class}, Confidence: {confidence}%")
    
    # Display class information
    st.write(f"Class Information:")
    st.write(f"Emoji: {class_info[predicted_class]['emoji']}")
    st.write(f"Description: {class_info[predicted_class]['description']}")
    st.write(f"Management: {class_info[predicted_class]['management']}")
