import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def main():
    # Load configuration settings from the .env file
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')

    if not ai_endpoint or not ai_key:
        st.error("AI service endpoint or key is not set in the .env file")
        return

    # Authenticate Azure AI Vision client
    cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
    )

    st.title("Azure AI Vision Text Reader")
    st.write("Upload an image to analyze the text using Azure AI Vision.")

    # File upload component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing...")

        # Perform text reading
        GetTextRead(cv_client, uploaded_file)


def GetTextRead(cv_client, image_file):
    # Open image file
    image_data = image_file.read()

    # Use Analyze image function to read text in image
    try:
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        # Display the image and overlay it with the extracted text
        if result.read is not None:
            st.write("Text found in the image:")

            image = Image.open(image_file)
            fig = plt.figure(figsize=(image.width / 100, image.height / 100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for block in result.read.blocks:
                for line in block.lines:
                    st.write(f"- {line.text}")

                    # Draw the bounding polygon for each line
                    if line.bounding_polygon:
                        r = line.bounding_polygon
                        bounding_polygon = [(r[i].x, r[i].y) for i in range(len(r))]
                        draw.polygon(bounding_polygon, outline=color, width=3)

            # Display the processed image with highlighted text
            st.image(image, caption="Processed Image with Text", use_column_width=True)
        else:
            st.write("No text found in the image.")

    except Exception as e:
        st.error(f"Error during text reading: {e}")


if __name__ == "__main__":
    main()
