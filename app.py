from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# Page config
st.set_page_config(page_title="Bone Age Prediction", layout="wide")

# Constants from training
IMG_SIZE = 299
CLASS_NAMES = ['Infant (0-2y)', 'Pre-Puberty (2-10y)', 'Puberty (10-16y)', 'Young Adult (16+y)']
BASE_DIR = Path(__file__).resolve().parent

DATASET_CSV_CANDIDATES = [
    BASE_DIR / "kaggle" / "input" / "rsna-bone-age" / "boneage-training-dataset.csv",
    BASE_DIR / "boneage-training-dataset.csv",
]

IMAGE_DIR_CANDIDATES = [
    BASE_DIR / "kaggle" / "input" / "rsna-bone-age" / "boneage-training-dataset" / "boneage-training-dataset",
    BASE_DIR / "kaggle" / "input" / "rsna-bone-age" / "boneage-training-dataset",
    BASE_DIR / "kaggle" / "input" / "rsna-bone-age" / "boneage-training-dataset" / "boneage-training-dataset" / "boneage-training-dataset",
]

TRAINING_HISTORY_PATH = BASE_DIR / "training_history.csv"

PRIMARY_PLOT_ARTIFACTS = [
    ("Training History", BASE_DIR / "training_history.png", "Train vs validation curves"),
    ("Regression Analysis", BASE_DIR / "regression_analysis.png", "Predicted vs actual ages and residuals"),
    ("Classification Confusion Matrix", BASE_DIR / "classification_confusion_matrix.png", "Counts and normalized confusion matrices"),
    ("Gender Bias Analysis", BASE_DIR / "gender_bias_analysis.png", "Regression/classification error split by gender"),
    ("Error by Age Group", BASE_DIR / "error_by_age_group.png", "MAE across developmental stages"),
]

SUPPLEMENTAL_PLOT_ARTIFACTS = [
    ("Calibration Results", BASE_DIR / "calibration_results.png", "Bias and isotonic calibration curves"),
    ("Uncertainty Analysis", BASE_DIR / "uncertainty_analysis.png", "Monte Carlo dropout uncertainty diagnostics"),
    ("Comprehensive Evaluation", BASE_DIR / "comprehensive_evaluation.png", "Full dashboard of residuals, confusion matrix, and gaps"),
    ("Model Architecture Diagram", BASE_DIR / "model_plot.png", "Keras model plot with layer details"),
]


def discretize_age(age_months: float) -> int:
    if age_months <= 24:
        return 0
    if age_months <= 120:
        return 1
    if age_months <= 192:
        return 2
    return 3


def resolve_dataset_csv() -> Optional[Path]:
    for candidate in DATASET_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def resolve_image_dir() -> Optional[Path]:
    for candidate in IMAGE_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_metadata(csv_path: Optional[str]):
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    df['age_class'] = df['boneage'].apply(discretize_age)
    df['age_stage'] = df['age_class'].map(dict(enumerate(CLASS_NAMES)))
    df['sex_label'] = df['male'].map({True: 'Male', False: 'Female'})
    return df


@st.cache_data(show_spinner=False)
def load_training_history(csv_path: Optional[str]):
    if not csv_path or not Path(csv_path).exists():
        return None
    history_df = pd.read_csv(csv_path)
    return history_df


def load_sample_images(metadata: Optional[pd.DataFrame], image_dir: Optional[Path], per_class: int = 1):
    if metadata is None or image_dir is None:
        return []
    samples = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        subset = metadata[metadata['age_class'] == class_idx]
        if subset.empty:
            continue
        take = min(per_class, len(subset))
        chosen = subset.sample(take, random_state=42)
        for _, row in chosen.iterrows():
            img_path = image_dir / f"{int(row['id'])}.png"
            if img_path.exists():
                samples.append({
                    'path': img_path,
                    'title': class_name,
                    'age': row['boneage'],
                    'sex': row['sex_label'],
                })
    return samples


def render_plot_gallery(items):
    available_items = [item for item in items if item[1].exists()]
    if not available_items:
        st.info("No saved visualizations were found in the workspace.")
        return
    for title, path, caption in available_items:
        with st.expander(title, expanded=False):
            if caption:
                st.caption(caption)
            st.image(str(path), use_container_width=True)
            with open(path, 'rb') as file:
                st.download_button(
                    label=f"Download {path.name}",
                    data=file.read(),
                    file_name=path.name,
                    mime="image/png",
                )

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('best_bone_age_model.keras')
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
    view = st.sidebar.radio(
        "Select View",
        [
            "Make Prediction",
            "Model Summary",
            "Dataset Explorer",
            "Training & Evaluation",
            "Visual Reports",
            "Model Architecture",
        ],
    )
    
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
            prob_df = pd.DataFrame(
                {
                    'Stage': CLASS_NAMES,
                    'Probability': class_probs,
                    'Percent': [f"{p*100:.1f}%" for p in class_probs],
                }
            )
            st.bar_chart(prob_df.set_index('Stage')['Probability'])
            st.dataframe(prob_df[['Stage', 'Percent']], use_container_width=True)
    
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
    
    elif view == "Dataset Explorer":
        st.header("Dataset Explorer")

        dataset_csv = resolve_dataset_csv()
        metadata = load_metadata(str(dataset_csv)) if dataset_csv else None
        image_dir = resolve_image_dir()

        if metadata is None:
            st.warning("Dataset CSV was not found. Place the RSNA CSV next to the app or under kaggle/input.")
        else:
            total_samples = len(metadata)
            male_pct = metadata['male'].mean() * 100
            age_min = metadata['boneage'].min()
            age_max = metadata['boneage'].max()
            age_mean = metadata['boneage'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", f"{total_samples:,}")
            col2.metric("Age Range (months)", f"{age_min:.0f} – {age_max:.0f}")
            col3.metric("Mean Age", f"{age_mean:.1f}")

            col4, col5 = st.columns(2)
            col4.metric("Gender Split (Male)", f"{male_pct:.1f}%")
            col5.metric("Gender Split (Female)", f"{100 - male_pct:.1f}%")

            st.subheader("Age Distribution")
            fig_age = px.histogram(
                metadata,
                x="boneage",
                nbins=60,
                color="sex_label",
                marginal="box",
                labels={"boneage": "Bone Age (months)", "sex_label": "Sex"},
                color_discrete_sequence=["#1f77b4", "#e377c2"],
            )
            st.plotly_chart(fig_age, use_container_width=True)

            class_counts = (
                metadata['age_stage']
                .value_counts()
                .reindex(CLASS_NAMES)
                .fillna(0)
                .rename_axis('Stage')
                .reset_index(name='Count')
            )
            st.subheader("Developmental Stage Distribution")
            fig_class = px.bar(
                class_counts,
                x='Stage',
                y='Count',
                text='Count',
                color='Stage',
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_class.update_traces(textposition='outside')
            st.plotly_chart(fig_class, use_container_width=True)

            st.subheader("Sample X-rays per Stage")
            samples = load_sample_images(metadata, image_dir)
            if not samples:
                st.info("Sample images are unavailable. Ensure the Kaggle image folder is present.")
            else:
                cols = st.columns(len(samples))
                for col, sample in zip(cols, samples):
                    caption = f"{sample['title']}\nAge: {sample['age']:.0f} months | Sex: {sample['sex']}"
                    col.image(str(sample['path']), caption=caption, use_container_width=True)

    elif view == "Training & Evaluation":
        st.header("Training & Evaluation Insights")

        history_df = load_training_history(str(TRAINING_HISTORY_PATH) if TRAINING_HISTORY_PATH.exists() else None)
        if history_df is None or history_df.empty:
            st.warning("training_history.csv not found. Run training to log history.")
        else:
            history_df = history_df.copy()
            if 'epoch' in history_df.columns:
                history_df['epoch_idx'] = history_df['epoch']
            else:
                history_df['epoch_idx'] = np.arange(len(history_df))

            final_row = history_df.iloc[-1]
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Val MAE", f"{final_row.get('val_age_out_mae', np.nan):.2f}", help="Validation MAE in months")
            metrics_cols[1].metric("Val RMSE", f"{np.sqrt(final_row.get('val_age_out_mse', np.nan)):.2f}")
            metrics_cols[2].metric("Val Accuracy", f"{final_row.get('val_class_out_accuracy', 0)*100:.2f}%")
            metrics_cols[3].metric("Val Loss", f"{final_row.get('val_loss', np.nan):.2f}")

            st.subheader("Learning Curves")
            mae_fig = px.line(
                history_df,
                x='epoch_idx',
                y=['val_age_out_mae'],
                labels={'value': 'MAE (months)', 'variable': 'Series', 'epoch_idx': 'Epoch'},
            )
            mae_fig.update_layout(showlegend=False)
            acc_fig = px.line(
                history_df,
                x='epoch_idx',
                y=['val_class_out_accuracy'],
                labels={'value': 'Accuracy', 'variable': 'Metric', 'epoch_idx': 'Epoch'},
            )
            acc_fig.update_layout(showlegend=False)
            col_chart1, col_chart2 = st.columns(2)
            col_chart1.plotly_chart(mae_fig, use_container_width=True)
            col_chart2.plotly_chart(acc_fig, use_container_width=True)

            st.subheader("Recent Training Logs")
            st.dataframe(history_df.tail(5))

            st.subheader("Key Evaluation Figures")
            render_plot_gallery(PRIMARY_PLOT_ARTIFACTS)

    elif view == "Visual Reports":
        st.header("Visual Reports & Diagnostics")
        st.write("Browse every saved PNG artifact without leaving Streamlit.")
        render_plot_gallery(PRIMARY_PLOT_ARTIFACTS + SUPPLEMENTAL_PLOT_ARTIFACTS)

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
