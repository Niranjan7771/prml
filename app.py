import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Bone Age Prediction", layout="wide")

# Constants from training
IMG_SIZE = 299
CLASS_NAMES = ['Infant (0-2y)', 'Pre-Puberty (2-10y)', 'Puberty (10-16y)', 'Young Adult (16+y)']

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('best_boneage_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for InceptionV3 (scale to [-1, 1])"""
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    # InceptionV3 preprocessing: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

def get_layer_output_shape(layer):
    try:
        return str(layer.output_shape)
    except:
        return "N/A"

def main():
    st.title("🦴 Bone Age Prediction from Hand Radiographs")
    st.markdown("**Multi-Task Learning: Regression + Classification**")
    
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigation")
    view = st.sidebar.radio("Select View", ["Make Prediction", "Model Summary", "Model Architecture"])
    
    if view == "Make Prediction":
        st.header("Predict Bone Age")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            uploaded_file = st.file_uploader("Upload Hand X-ray Image", type=['png', 'jpg', 'jpeg'])
            gender = st.radio("Patient Gender", ["Male", "Female"])
            gender_value = np.array([[1.0 if gender == "Male" else 0.0]], dtype=np.float32)
        
        with col2:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        if uploaded_file is not None:
            st.divider()
            st.subheader("Prediction Results")
            
            # Preprocess and predict
            img_array = preprocess_image(image)
            
            with st.spinner("Analyzing X-ray..."):
                predictions = model.predict([img_array, gender_value], verbose=0)
            
            # Model outputs: [age_out, class_out]
            age_pred = predictions[0][0][0]  # Regression output (months)
            class_probs = predictions[1][0]  # Classification probabilities
            class_pred = np.argmax(class_probs)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Bone Age", f"{age_pred:.1f} months")
                st.write(f"≈ {age_pred/12:.1f} years")
            
            with col2:
                st.metric("Developmental Stage", CLASS_NAMES[class_pred])
                st.write(f"Confidence: {class_probs[class_pred]*100:.1f}%")
            
            with col3:
                st.metric("Gender Input", gender)
            
            # Class probabilities
            st.subheader("Classification Probabilities")
            prob_data = {CLASS_NAMES[i]: f"{class_probs[i]*100:.1f}%" for i in range(len(CLASS_NAMES))}
            st.bar_chart({CLASS_NAMES[i]: class_probs[i] for i in range(len(CLASS_NAMES))})
    
    elif view == "Model Summary":
        st.header("Model Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.write(f"**Architecture:** InceptionV3 + Multi-Task Heads")
            st.write(f"**Total Layers:** {len(model.layers)}")
            st.write(f"**Input 1:** Image (299×299×3)")
            st.write(f"**Input 2:** Gender (1)")
            st.write(f"**Output 1:** Age in months (regression)")
            st.write(f"**Output 2:** Age class (4 categories)")
            
            trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
            non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
            st.write(f"**Trainable Params:** {trainable:,}")
            st.write(f"**Non-trainable Params:** {non_trainable:,}")
            st.write(f"**Total Params:** {trainable + non_trainable:,}")
        
        with col2:
            st.subheader("Classification Categories")
            for i, name in enumerate(CLASS_NAMES):
                st.write(f"**Class {i}:** {name}")
        
        st.subheader("Layer Details")
        layer_data = []
        for i, layer in enumerate(model.layers[:50]):  # First 50 layers
            layer_data.append({
                "Index": i,
                "Name": layer.name,
                "Type": layer.__class__.__name__,
                "Output Shape": get_layer_output_shape(layer)
            })
        st.dataframe(layer_data, use_container_width=True)
        
        st.subheader("Full Model Summary")
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        st.text('\n'.join(summary_lines))
    
    elif view == "Model Architecture":
        st.header("Model Architecture")
        
        try:
            plot_path = "model_architecture.png"
            keras.utils.plot_model(
                model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=96
            )
            st.image(plot_path, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate plot: {e}")
            st.info("Install graphviz and pydot: `pip install pydot graphviz`")

if __name__ == "__main__":
    main()
