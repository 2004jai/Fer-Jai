import streamlit as st
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

def main():
    st.set_page_config(
        page_title="Emotion Detection App",
        page_icon="😀",
        layout="wide"
    )
    
    st.title("Real-Time Emotion Detection")
    st.sidebar.title("Settings")
    
    # App modes
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["About", "Run on Video", "Upload Image"]
    )
    
    if app_mode == "About":
        show_about_page()
    elif app_mode == "Run on Video":
        run_on_video()
    elif app_mode == "Upload Image":
        run_on_image()

def show_about_page():
    st.markdown("""
    ## About This App
    
    This application uses deep learning to detect emotions in real-time from your webcam feed or from uploaded images.
    
    ### How it works
    1. Face detection using dlib's HOG-based face detector
    2. Facial landmark extraction
    3. Face preprocessing (grayscale conversion, histogram equalization)
    4. Emotion classification using a pre-trained Convolutional Neural Network
    
    ### Detected Emotions
    - Angry
    - Disgust
    - Fear
    - Happy
    - Sad
    - Surprise
    - Neutral
    
    ### Tech Stack
    - Streamlit for the web interface
    - OpenCV for image processing
    - dlib for face detection
    - TensorFlow/Keras for the emotion classification model
    """)

@st.cache_resource
def load_detection_model():
    # Load face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        st.error(f"Error: {predictor_path} not found. Please download it and place it in the app directory.")
        st.stop()
        
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

@st.cache_resource
def load_emotion_model():
    # Load pre-trained emotion model
    model_path = "emotion.h5"
    
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found. Please download it and place it in the app directory.")
        st.stop()
        
    model = load_model(model_path)
    return model

def preprocess_image(image):
    # Check if the image is empty
    if image is None or image.size == 0:
        return None
    
    # Check if the image is already grayscale (1 channel)
    if len(image.shape) == 2:  # Grayscale image
        pass  # No need to convert
    else:  # BGR image (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    # Resize to 48x48
    image = cv2.resize(image, (48, 48))
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def process_image(image, detector, predictor, model, emotion_labels):
    # Convert to grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = detector(gray)
    
    # Create a copy of the image to draw on
    result_image = image.copy()
    
    # List to store emotion data for each face
    face_emotions = []
    
    for i, face in enumerate(faces):
        # Get face landmarks
        landmarks = predictor(gray, face)
        
        # Get face bounding box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extract face region
        face_image = gray[y:y+h, x:x+w]
        
        # Skip if face image is empty
        if face_image.size == 0:
            continue
        
        # Preprocess face for emotion detection
        processed_image = preprocess_image(face_image)
        if processed_image is None:
            continue
        
        # Predict emotion
        prediction = model.predict(processed_image)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        
        # Draw rectangle around face
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw emotion label above face
        cv2.putText(result_image, f"{emotion}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Store emotion data for this face
        face_data = {
            "face_id": i+1,
            "emotion": emotion,
            "probabilities": {label: float(prob) for label, prob in zip(emotion_labels, prediction[0])}
        }
        face_emotions.append(face_data)
    
    return result_image, face_emotions

def run_on_image():
    # Load models
    detector, predictor = load_detection_model()
    model = load_emotion_model()
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="RGB", use_column_width=True)
        
        # Process image
        result_image, face_emotions = process_image(image, detector, predictor, model, emotion_labels)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detected Emotions")
            st.image(result_image, channels="RGB", use_column_width=True)
        
        with col2:
            st.subheader("Emotion Analysis")
            if len(face_emotions) == 0:
                st.write("No faces detected in the image.")
            else:
                st.write(f"Detected {len(face_emotions)} face(s)")
                
                for face_data in face_emotions:
                    st.write(f"### Face #{face_data['face_id']}: {face_data['emotion']}")
                    
                    # Create a bar chart of emotion probabilities
                    probabilities = face_data["probabilities"]
                    chart_data = {
                        "Emotion": list(probabilities.keys()),
                        "Probability": list(probabilities.values())
                    }
                    st.bar_chart(chart_data, x="Emotion", y="Probability", use_container_width=True)

def run_on_video():
    # Load models
    detector, predictor = load_detection_model()
    model = load_emotion_model()
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    st.sidebar.markdown("## Video Settings")
    
    # Video source selection
    video_source = st.sidebar.radio("Select video source", ["Webcam", "Upload Video"])
    
    if video_source == "Webcam":
        # Start webcam
        cap = cv2.VideoCapture(0)
        
        # Stop button
        stop_button = st.sidebar.button("Stop")
        
        # Webcam feed placeholder
        stframe = st.empty()
        
        # Video settings
        st.sidebar.markdown("## Display Settings")
        display_score = st.sidebar.checkbox("Display Emotion Scores", value=True)
        
        # Analytics containers
        face_count_container = st.sidebar.empty()
        emotion_stats_container = st.sidebar.empty()
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            
            if ret:
                # Convert frame to RGB (from BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result_frame, face_emotions = process_image(frame, detector, predictor, model, emotion_labels)
                
                # Update face count
                face_count_container.markdown(f"### Detected Faces: {len(face_emotions)}")
                
                # Update emotion stats
                if len(face_emotions) > 0:
                    emotion_counts = {}
                    for face_data in face_emotions:
                        emotion = face_data["emotion"]
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    emotion_stats = "\n".join([f"- {emotion}: {count}" for emotion, count in emotion_counts.items()])
                    emotion_stats_container.markdown(f"### Emotions Detected:\n{emotion_stats}")
                
                # Display frame
                stframe.image(result_frame, channels="RGB", use_column_width=True)
            else:
                st.write("Error: Unable to read from webcam.")
                break
        
        # Release webcam
        cap.release()
        
    elif video_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Read video file
            cap = cv2.VideoCapture(tfile.name)
            
            # Video controls
            st.sidebar.markdown("## Video Controls")
            start_button = st.sidebar.button("Start Analysis")
            stop_button = st.sidebar.button("Stop Analysis")
            
            # Video settings
            st.sidebar.markdown("## Display Settings")
            display_score = st.sidebar.checkbox("Display Emotion Scores", value=True)
            
            # Video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            
            st.sidebar.markdown(f"**Video Info:**")
            st.sidebar.markdown(f"- Duration: {duration:.2f} seconds")
            st.sidebar.markdown(f"- Frames: {total_frames}")
            st.sidebar.markdown(f"- FPS: {fps}")
            
            if start_button:
                # Video frame placeholder
                stframe = st.empty()
                
                # Progress bar
                progress_bar = st.progress(0)
                
                # Analytics containers
                face_count_container = st.sidebar.empty()
                emotion_stats_container = st.sidebar.empty()
                
                # Process video frames
                frame_index = 0
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert frame to RGB (from BGR)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process frame
                        result_frame, face_emotions = process_image(frame, detector, predictor, model, emotion_labels)
                        
                        # Update face count
                        face_count_container.markdown(f"### Detected Faces: {len(face_emotions)}")
                        
                        # Update emotion stats
                        if len(face_emotions) > 0:
                            emotion_counts = {}
                            for face_data in face_emotions:
                                emotion = face_data["emotion"]
                                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                            
                            emotion_stats = "\n".join([f"- {emotion}: {count}" for emotion, count in emotion_counts.items()])
                            emotion_stats_container.markdown(f"### Emotions Detected:\n{emotion_stats}")
                        
                        # Display frame
                        stframe.image(result_frame, channels="RGB", use_column_width=True)
                        
                        # Update progress
                        frame_index += 1
                        progress_bar.progress(min(frame_index / total_frames, 1.0))
                    else:
                        break
                
                # Release video
                cap.release()
                
                # Remove temporary file
                os.unlink(tfile.name)

if __name__ == "__main__":
    main()