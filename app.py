import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = load_model('cnn_model.h5')

# Function to process the uploaded image
def process_image(img):
    img = img.convert('RGB')  
    img = img.resize((32, 32)) 
    img = np.array(img)  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img

# Frontend design
st.set_page_config(page_title="Dog vs Cat Detection", page_icon="🐶🐱", layout="centered")
st.title("Dog vs Cat Image Classification 🐶🐱")

# Description
st.markdown("""
    This is a simple Dog vs Cat image classifier. Upload an image of either a dog or a cat, and 
    the model will predict the class along with the confidence level.
    """)

# Image upload
file = st.file_uploader('Select an image', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    
    # Display the uploaded image with a border and centered
    st.image(img, caption='Uploaded Image', use_column_width=True, 
             output_format="PNG", width=400)

    # Preprocess the image
    image = process_image(img)
    
    # Model prediction
    with st.spinner('Classifying the image...'):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)  
        predicted_prob = predictions[0][predicted_class]  

    # Class names for prediction
    class_names = ['Cat','Dog']

    # Display the prediction result
    st.subheader(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {predicted_prob * 100:.2f}%")

    # Display prediction probabilities
    st.write("Prediction Probabilities for Each Class:")

    # Prepare probabilities for visualization
    probabilities = predictions[0]
    prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
    
    # Plot settings
    sns.set(style="whitegrid")  # Use a grid style for the plot

    # Create the figure for the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability

    # Plot the bar chart with a brighter color palette
    ax.bar(list(prob_dict.keys()), list(prob_dict.values()), color='#f5a623', edgecolor='black')
    ax.set_ylabel('Probability', fontsize=14, color='black')
    ax.set_title('Prediction Probabilities for Each Class', fontsize=18, color='black')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Annotate bars with percentage values
    for index, value in enumerate(prob_dict.values()):
        ax.text(index, value, f'{value * 100:.0f}%', va='bottom', ha='center', fontsize=10)

    # Style improvements: Remove background grid and spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    # Adjust layout to prevent clipping
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

