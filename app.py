import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import requests  # For calling the Video Indexer REST API

# Import namespaces for Azure Vision and Text Analytics
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Initialize the Azure clients globally
cv_client = None
text_client = None
video_indexer_key = None
video_indexer_endpoint = None
video_indexer_location = None

def main():
    global cv_client, text_client, video_indexer_key, video_indexer_endpoint, video_indexer_location

    # Load environment variables
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')
    text_endpoint = os.getenv('TEXT_ANALYTICS_ENDPOINT')
    text_key = os.getenv('TEXT_ANALYTICS_KEY')
    video_indexer_key = os.getenv('VIDEO_INDEXER_API_KEY')
    video_indexer_endpoint = os.getenv('VIDEO_INDEXER_ENDPOINT')
    video_indexer_location = os.getenv('VIDEO_INDEXER_LOCATION')

    # Validate endpoints and keys
    if not ai_endpoint or not ai_key or not text_endpoint or not text_key or not video_indexer_key or not video_indexer_endpoint:
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

    st.title("Azure AI Vision and Video Indexer with Sentiment Analysis")
    st.write("Upload an image or video to analyze text and extract insights.")

    # File upload component
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])
    camera_input = st.camera_input("Take a picture...")

    if uploaded_file is not None:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image_data = uploaded_file.read()
            process_image(image_data, uploaded_file)
        elif uploaded_file.type == "video/mp4":
            video_data = uploaded_file.read()
            process_video(video_data)
    elif camera_input is not None:
        image_data = camera_input.read()
        process_image(image_data, camera_input)
    else:
        st.write("Please upload an image or video, or take a picture.")

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

def process_video(video_data):
    st.write("Processing video...")
    video_url = upload_video_to_video_indexer(video_data)
    if video_url:
        video_insights = get_video_insights(video_url)
        st.write("Video analysis complete!")
        st.write(video_insights)

def upload_video_to_video_indexer(video_data):
    try:
        # Upload the video to Azure Video Indexer
        url = f"https://{video_indexer_location}.api.videoindexer.ai/{video_indexer_location}/Accounts/{video_indexer_key}/Videos"
        headers = {
            'Ocp-Apim-Subscription-Key': video_indexer_key,
        }
        files = {
            'file': ('video.mp4', video_data, 'video/mp4'),
        }

        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()

        result = response.json()
        video_id = result['id']
        st.write(f"Video uploaded successfully: {video_id}")
        return video_id

    except Exception as e:
        st.error(f"Error uploading video: {e}")
        return None

def get_video_insights(video_id):
    try:
        # Retrieve the video insights
        url = f"https://{video_indexer_location}.api.videoindexer.ai/{video_indexer_location}/Accounts/{video_indexer_key}/Videos/{video_id}/Index"
        headers = {
            'Ocp-Apim-Subscription-Key': video_indexer_key,
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        insights = response.json()
        return insights

    except Exception as e:
        st.error(f"Error retrieving video insights: {e}")
        return None

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
