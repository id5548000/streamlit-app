import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces for Azure Vision and Text Analytics
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Initialize the Azure clients globally
cv_client = None
text_client = None

def main():
    global cv_client, text_client

    # Load environment variables
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')
    text_endpoint = os.getenv('TEXT_ANALYTICS_ENDPOINT')
    text_key = os.getenv('TEXT_ANALYTICS_KEY')

    # Validate endpoints and keys
    if not ai_endpoint or not ai_key or not text_endpoint or not text_key:
        raise ValueError("One or more service endpoints or keys are not set in the .env file")

    # Authenticate Azure clients
    cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
    )

    text_client = TextAnalyticsClient(
        endpoint=text_endpoint,
        credential=AzureKeyCredential(text_key)
    )

    st.title("Azure AI Vision Text Reader with Sentiment Analysis")
    st.write("Upload an image or take a picture to analyze the text and get sentiment analysis.")

    # File upload component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    camera_input = st.camera_input("Take a picture...")

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        process_image(image_data, uploaded_file)
    elif camera_input is not None:
        image_data = camera_input.read()
        process_image(image_data, camera_input)
    else:
        st.write("Please upload an image or take a picture.")

def process_image(image_data, image_file):
    # Debug: Show the image size
    image_data_size = len(image_data)
    st.write(f"Image size: {image_data_size} bytes")

    # Validate image size
    if image_data_size == 0:
        st.error("Uploaded image is empty. Please upload a valid image.")
        return
    elif image_data_size > 20971520:  # 20 MB in bytes
        st.error("Uploaded image is too large. Please upload an image smaller than 20 MB.")
        return

    # Display the uploaded image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    # Perform text reading
    extracted_text = GetTextRead(image_file, image_data, image)
    if extracted_text:
        sentiment_analysis(extracted_text)

def GetTextRead(image_file, image_data, original_image):
    try:
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        if result.read is not None:
            extracted_text = ""
            draw = ImageDraw.Draw(original_image)
            color = 'cyan'

            for block in result.read.blocks:
                for line in block.lines:
                    extracted_text += f"{line.text} "
                    # Draw bounding box
                    if line.bounding_polygon:
                        r = line.bounding_polygon
                        bounding_polygon = [(r[i].x, r[i].y) for i in range(len(r))]
                        draw.polygon(bounding_polygon, outline=color, width=3)

                        # Overlay the text on the image
                        draw.text((bounding_polygon[0][0], bounding_polygon[0][1]), line.text, fill=color)

            st.write("Text found in the image:")
            st.write(extracted_text.strip())

            # Display the image with overlays
            st.image(original_image, caption="Processed Image with Text", use_column_width=True)
            return extracted_text.strip()
        else:
            st.write("No text found in the image.")
            return None

    except Exception as e:
        st.error(f"Error during text reading: {e}")
        return None

def sentiment_analysis(text):
    try:
        response = text_client.analyze_sentiment([text])[0]
        st.write("Sentiment Analysis Result:")
        st.write(f"Document sentiment: {response.sentiment}")
        for sentence in response.sentences:
            st.write(f"- Sentence: {sentence.text}")
            st.write(f"  Sentiment: {sentence.sentiment}, Confidence scores: {sentence.confidence_scores}")
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")

if __name__ == "__main__":
    main()
