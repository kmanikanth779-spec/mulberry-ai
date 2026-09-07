import tensorflow as tf
import numpy as np
import cv2

# -------------------------------
# GRAD-CAM HEATMAP GENERATION
# -------------------------------
def get_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="top_conv"
):
    """
    Generates a smooth Grad-CAM heatmap normalized to [0,1]
    """
    try:
        # Create grad model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_class]

        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)

        # Channel-wise importance
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the convolution outputs
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap.numpy().astype(np.float32)

    except Exception as e:
        print(f"❌ Grad-CAM Error: {e}")
        return None


# -------------------------------
# OVERLAY HEATMAP ON IMAGE
# -------------------------------
def overlay_gradcam(
    image,
    heatmap,
    alpha=0.45,
    colormap=cv2.COLORMAP_JET
):
    """
    Overlays Grad-CAM heatmap on original image
    """
    if heatmap is None:
        return image

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert to 0–255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply color map (JET = red/yellow focus)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert image to BGR if needed
    if image.shape[-1] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    # Overlay
    overlay = cv2.addWeighted(
        image_bgr, 1 - alpha,
        heatmap_color, alpha,
        0
    )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# -------------------------------
# SEVERITY CALCULATION
# -------------------------------
def calculate_severity_percentage(heatmap, threshold=0.25):
    """
    Calculates % area affected and severity label
    """
    if heatmap is None:
        return 0.0, "Unknown", "⚪"

    diseased_pixels = np.sum(heatmap > threshold)
    total_pixels = heatmap.size
    severity_pct = (diseased_pixels / total_pixels) * 100

    if severity_pct < 15:
        return severity_pct, "Low", "🟢"
    elif severity_pct < 35:
        return severity_pct, "Medium", "🟡"
    else:
        return severity_pct, "High", "🔴"
