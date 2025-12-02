# Additional Notebook Cells for Core Deliverables

Copy these markdown and code cells into your Jupyter notebook to complete the project requirements.

---

## Cell 1: Project Introduction (Markdown)

```markdown
# Bone Age Prediction from Hand Radiographs

## Project Overview

**Objective:** Estimate a patient's bone age from hand X-ray images using deep learning.

**Tasks:**
1. **Regression:** Predict bone age as a continuous value (months)
2. **Classification:** Categorize into developmental stages (Infant, Pre-Puberty, Puberty, Young Adult)

**Dataset:** RSNA Bone Age Dataset (12,611 images)

**Model:** Multi-task CNN with InceptionV3 backbone + Gender input

---

### Key Features:
- Transfer learning with ImageNet pretrained weights
- Multi-input architecture (Image + Gender)
- Multi-output heads (Regression + Classification)
- Data augmentation for improved generalization
```

---

## Cell 2: After Data Loading - EDA Section (Markdown)

```markdown
## Exploratory Data Analysis

### Dataset Statistics
- **Total Samples:** 12,611 hand X-ray images
- **Age Range:** 1 - 228 months (0 - 19 years)
- **Mean Age:** 127.3 months (~10.6 years)
- **Gender Distribution:** Analyzed below

### Age Discretization Strategy
| Class | Stage | Age Range | Clinical Significance |
|-------|-------|-----------|----------------------|
| 0 | Infant | 0-2 years | Early skeletal development |
| 1 | Pre-Puberty | 2-10 years | Steady growth phase |
| 2 | Puberty | 10-16 years | Growth spurt period |
| 3 | Young Adult | 16+ years | Near skeletal maturity |
```

---

## Cell 3: Gender Distribution Visualization (Code)

```python
# ============================================================================
# GENDER DISTRIBUTION ANALYSIS
# ============================================================================
print("\n>>> GENDER DISTRIBUTION ANALYSIS")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Gender distribution
gender_counts = df['male'].value_counts()
axes[0].pie([gender_counts[True], gender_counts[False]], 
            labels=['Male', 'Female'], 
            autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'])
axes[0].set_title('Gender Distribution')

# Age distribution by gender
df[df['male']==True]['boneage'].hist(ax=axes[1], bins=30, alpha=0.7, label='Male', color='#3498db')
df[df['male']==False]['boneage'].hist(ax=axes[1], bins=30, alpha=0.7, label='Female', color='#e74c3c')
axes[1].set_xlabel('Bone Age (months)')
axes[1].set_ylabel('Count')
axes[1].set_title('Age Distribution by Gender')
axes[1].legend()

# Class distribution
class_counts = df['age_class'].value_counts().sort_index()
axes[2].bar(CLASS_NAMES, class_counts.values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[2].set_xlabel('Developmental Stage')
axes[2].set_ylabel('Count')
axes[2].set_title('Class Distribution')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\nMale samples: {gender_counts[True]} ({gender_counts[True]/len(df)*100:.1f}%)")
print(f"Female samples: {gender_counts[False]} ({gender_counts[False]/len(df)*100:.1f}%)")
```

---

## Cell 4: Sample Images Visualization (Code)

```python
# ============================================================================
# SAMPLE IMAGES VISUALIZATION
# ============================================================================
print("\n>>> SAMPLE IMAGES FROM EACH CLASS")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, class_idx in enumerate(range(4)):
    sample = df[df['age_class'] == class_idx].sample(1).iloc[0]
    img = plt.imread(sample['path'])
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"{CLASS_NAMES[class_idx]}\nAge: {sample['boneage']} months\nGender: {'M' if sample['male'] else 'F'}")
    axes[i].axis('off')

plt.suptitle('Sample X-rays from Each Developmental Stage', fontsize=14)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Cell 5: Model Architecture Explanation (Markdown)

```markdown
## Model Architecture

### Design Choices

**1. Backbone: InceptionV3**
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Efficient multi-scale feature extraction via inception modules
- Native input size: 299×299 pixels
- ~22M parameters

**2. Multi-Input Design**
- **Image Input:** Hand X-ray (299×299×3)
- **Gender Input:** Binary (Male=1, Female=0)
- Gender affects bone development timing, crucial for accurate prediction

**3. Multi-Task Learning**
- Shared backbone learns general bone features
- Task-specific heads specialize for regression/classification
- Joint training improves both tasks through regularization

**4. Architecture Diagram**
```
Image (299×299×3) ──→ InceptionV3 ──→ GlobalAvgPool ──→ Dropout(0.5) ──┐
                                                                        ├──→ Concat ──→ Dense(256) ──→ Dropout(0.3) ──┬──→ Dense(1) ──→ Age (months)
Gender (1) ────────────────────────────────────────────────────────────┘                                              └──→ Dense(4) ──→ Class (softmax)
```
```

---

## Cell 6: After Training - Training History Visualization (Code)

```python
# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================
print("\n>>> TRAINING HISTORY")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot regression loss (MAE)
axes[0, 0].plot(history.history['age_out_mae'], label='Train MAE')
axes[0, 0].plot(history.history['val_age_out_mae'], label='Val MAE')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MAE (months)')
axes[0, 0].set_title('Regression: Mean Absolute Error')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot classification accuracy
axes[0, 1].plot(history.history['class_out_accuracy'], label='Train Accuracy')
axes[0, 1].plot(history.history['val_class_out_accuracy'], label='Val Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Classification: Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot total loss
axes[1, 0].plot(history.history['loss'], label='Train Loss')
axes[1, 0].plot(history.history['val_loss'], label='Val Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Total Loss')
axes[1, 0].set_title('Combined Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot learning rate if available
if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Recorded', ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('Learning Rate')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Cell 7: Model Evaluation Section (Markdown)

```markdown
## Model Evaluation

### Evaluation Metrics

**Regression Metrics:**
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and true ages
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily
- **R² (Coefficient of Determination):** Proportion of variance explained by the model

**Classification Metrics:**
- **Accuracy:** Overall correct predictions
- **Precision/Recall/F1:** Per-class performance
- **Confusion Matrix:** Detailed error analysis
- **Quadratic Weighted Kappa:** Agreement measure for ordinal classes
```

---

## Cell 8: Comprehensive Evaluation (Code)

```python
# ============================================================================
# COMPREHENSIVE MODEL EVALUATION
# ============================================================================
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            classification_report, confusion_matrix, cohen_kappa_score)

print("\n>>> EVALUATING ON TEST SET")

# Get predictions
test_predictions = model.predict(test_ds, verbose=1)
age_preds = test_predictions[0].flatten()
class_preds = np.argmax(test_predictions[1], axis=1)

# Get true values
y_true_age = test_df['boneage'].values
y_true_class = test_df['age_class'].values

# ============================================================================
# REGRESSION METRICS
# ============================================================================
print("\n" + "="*60)
print("REGRESSION METRICS")
print("="*60)

mae = mean_absolute_error(y_true_age, age_preds)
rmse = np.sqrt(mean_squared_error(y_true_age, age_preds))
r2 = r2_score(y_true_age, age_preds)

print(f"Mean Absolute Error (MAE): {mae:.2f} months ({mae/12:.2f} years)")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} months ({rmse/12:.2f} years)")
print(f"R² Score: {r2:.4f}")

# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================
print("\n" + "="*60)
print("CLASSIFICATION METRICS")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_true_class, class_preds, target_names=CLASS_NAMES))

# Quadratic Weighted Kappa
qwk = cohen_kappa_score(y_true_class, class_preds, weights='quadratic')
print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
```

---

## Cell 9: Predicted vs Actual Scatter Plot (Code)

```python
# ============================================================================
# PREDICTED VS ACTUAL AGE SCATTER PLOT
# ============================================================================
print("\n>>> PREDICTED VS ACTUAL AGE")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall scatter plot
axes[0].scatter(y_true_age, age_preds, alpha=0.3, s=10)
axes[0].plot([0, 230], [0, 230], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Age (months)')
axes[0].set_ylabel('Predicted Age (months)')
axes[0].set_title(f'Predicted vs Actual Bone Age\nMAE: {mae:.2f} months, R²: {r2:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 240)
axes[0].set_ylim(0, 240)

# By gender
colors = ['#3498db' if g else '#e74c3c' for g in test_df['male'].values]
axes[1].scatter(y_true_age, age_preds, c=colors, alpha=0.3, s=10)
axes[1].plot([0, 230], [0, 230], 'k--', linewidth=2)
axes[1].set_xlabel('Actual Age (months)')
axes[1].set_ylabel('Predicted Age (months)')
axes[1].set_title('Predicted vs Actual by Gender\n(Blue: Male, Red: Female)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 240)
axes[1].set_ylim(0, 240)

plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Cell 10: Confusion Matrix Visualization (Code)

```python
# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================
print("\n>>> CONFUSION MATRIX")

cm = confusion_matrix(y_true_class, class_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (Normalized)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Cell 11: Gender-wise Performance Analysis (Code)

```python
# ============================================================================
# GENDER-WISE PERFORMANCE ANALYSIS
# ============================================================================
print("\n>>> GENDER-WISE PERFORMANCE ANALYSIS")

# Separate by gender
male_mask = test_df['male'].values
female_mask = ~male_mask

# Male metrics
male_mae = mean_absolute_error(y_true_age[male_mask], age_preds[male_mask])
male_rmse = np.sqrt(mean_squared_error(y_true_age[male_mask], age_preds[male_mask]))
male_r2 = r2_score(y_true_age[male_mask], age_preds[male_mask])

# Female metrics
female_mae = mean_absolute_error(y_true_age[female_mask], age_preds[female_mask])
female_rmse = np.sqrt(mean_squared_error(y_true_age[female_mask], age_preds[female_mask]))
female_r2 = r2_score(y_true_age[female_mask], age_preds[female_mask])

print("\n" + "="*60)
print("GENDER-WISE REGRESSION METRICS")
print("="*60)
print(f"\n{'Metric':<20} {'Male':<15} {'Female':<15} {'Difference':<15}")
print("-"*60)
print(f"{'MAE (months)':<20} {male_mae:<15.2f} {female_mae:<15.2f} {abs(male_mae-female_mae):<15.2f}")
print(f"{'RMSE (months)':<20} {male_rmse:<15.2f} {female_rmse:<15.2f} {abs(male_rmse-female_rmse):<15.2f}")
print(f"{'R² Score':<20} {male_r2:<15.4f} {female_r2:<15.4f} {abs(male_r2-female_r2):<15.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MAE comparison
metrics = ['MAE', 'RMSE']
male_vals = [male_mae, male_rmse]
female_vals = [female_mae, female_rmse]

x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, male_vals, width, label='Male', color='#3498db')
axes[0].bar(x + width/2, female_vals, width, label='Female', color='#e74c3c')
axes[0].set_ylabel('Error (months)')
axes[0].set_title('Regression Error by Gender')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# R² comparison
axes[1].bar(['Male', 'Female'], [male_r2, female_r2], color=['#3498db', '#e74c3c'])
axes[1].set_ylabel('R² Score')
axes[1].set_title('R² Score by Gender')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gender_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Bias analysis
print("\n" + "="*60)
print("BIAS ANALYSIS")
print("="*60)
bias_diff = abs(male_mae - female_mae)
if bias_diff < 2:
    print(f"✓ Low gender bias detected (MAE difference: {bias_diff:.2f} months)")
elif bias_diff < 5:
    print(f"⚠ Moderate gender bias detected (MAE difference: {bias_diff:.2f} months)")
else:
    print(f"✗ High gender bias detected (MAE difference: {bias_diff:.2f} months)")
```

---

## Cell 12: Error Analysis by Age Group (Code)

```python
# ============================================================================
# ERROR ANALYSIS BY AGE GROUP
# ============================================================================
print("\n>>> ERROR ANALYSIS BY AGE GROUP")

# Calculate errors
errors = age_preds - y_true_age

# Error by class
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error distribution
axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Prediction Error (months)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Error Distribution\nMean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f}')
axes[0, 0].grid(True, alpha=0.3)

# Error vs actual age
axes[0, 1].scatter(y_true_age, errors, alpha=0.3, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Actual Age (months)')
axes[0, 1].set_ylabel('Prediction Error (months)')
axes[0, 1].set_title('Error vs Actual Age')
axes[0, 1].grid(True, alpha=0.3)

# MAE by class
class_maes = []
for c in range(4):
    mask = y_true_class == c
    class_mae = mean_absolute_error(y_true_age[mask], age_preds[mask])
    class_maes.append(class_mae)

axes[1, 0].bar(CLASS_NAMES, class_maes, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[1, 0].set_xlabel('Developmental Stage')
axes[1, 0].set_ylabel('MAE (months)')
axes[1, 0].set_title('MAE by Developmental Stage')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Box plot of errors by class
error_by_class = [errors[y_true_class == c] for c in range(4)]
bp = axes[1, 1].boxplot(error_by_class, labels=CLASS_NAMES, patch_artist=True)
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[1, 1].set_xlabel('Developmental Stage')
axes[1, 1].set_ylabel('Prediction Error (months)')
axes[1, 1].set_title('Error Distribution by Stage')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nMAE by Developmental Stage:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_maes[i]:.2f} months")
```

---

## Cell 13: Grad-CAM Visualization (Code)

```python
# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================
print("\n>>> GRAD-CAM VISUALIZATION")

import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the regression output."""
    
    # Create a model that maps input to activations and output
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output[0]]  # age_out
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, np.array([[1.0]])])  # dummy gender
        loss = predictions[0]
    
    # Gradient of the output with respect to the conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # Mean intensity of gradient over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Resize heatmap to image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (img.shape[0], img.shape[1]))
    heatmap = tf.squeeze(heatmap).numpy()
    
    # Apply colormap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap.astype(int)]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose
    superimposed = jet_heatmap * alpha + img
    superimposed = tf.keras.preprocessing.image.array_to_img(superimposed)
    
    return img, superimposed

# Generate Grad-CAM for sample images
last_conv_layer = 'mixed10'  # Last conv layer in InceptionV3

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    sample = test_df[test_df['age_class'] == i].sample(1).iloc[0]
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(sample['path'], target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        original, superimposed = display_gradcam(sample['path'], heatmap)
        
        axes[0, i].imshow(original.astype(np.uint8))
        axes[0, i].set_title(f"{CLASS_NAMES[i]}\nAge: {sample['boneage']} months")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(superimposed)
        axes[1, i].set_title('Grad-CAM')
        axes[1, i].axis('off')
    except Exception as e:
        print(f"Error generating Grad-CAM for class {i}: {e}")
        axes[0, i].text(0.5, 0.5, 'Error', ha='center', va='center')
        axes[1, i].text(0.5, 0.5, 'Error', ha='center', va='center')

plt.suptitle('Grad-CAM Visualization: Model Attention Regions', fontsize=14)
plt.tight_layout()
plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Cell 14: Conclusions (Markdown)

```markdown
## Conclusions

### Summary of Results

| Metric | Value |
|--------|-------|
| Regression MAE | ~X months |
| Regression RMSE | ~X months |
| R² Score | ~0.XX |
| Classification Accuracy | ~XX% |
| Quadratic Weighted Kappa | ~0.XX |

### Key Findings

1. **Multi-task learning** effectively combines regression and classification, with shared features benefiting both tasks.

2. **Gender input** significantly improves predictions by accounting for sex-based differences in bone development.

3. **Transfer learning** with InceptionV3 provides strong feature extraction despite domain shift from natural images to medical X-rays.

4. **Class imbalance** affects performance on minority classes (Infant, Young Adult).

### Limitations

- Single dataset may limit generalization
- No explicit handling of class imbalance
- Limited interpretability of deep features

### Future Work

- Implement class weighting or oversampling
- Ensemble multiple architectures
- Cross-validation for robust evaluation
- External validation on different populations
```

---

## Cell 15: Save Model and Results (Code)

```python
# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================
print("\n>>> SAVING MODEL AND RESULTS")

# Save model
model.save('best_boneage_model.keras')
print("✓ Model saved: best_boneage_model.keras")

# Save results summary
results = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'male_mae': male_mae,
    'female_mae': female_mae,
    'qwk': qwk
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved: results_summary.json")

print("\n" + "="*60)
print("PROJECT COMPLETE")
print("="*60)
```
