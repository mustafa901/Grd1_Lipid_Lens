import gradio as gr
import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def process_image(image):
    # Convert to NumPy array and ensure proper color format
    image = np.array(image, dtype=np.uint8)
    
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    # Define threshold values
    threshold_values = np.linspace(40, 300, num=50)
    
    # Create a figure for plotting results
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    axes = axes.flatten()

    for i, thresh_val in enumerate(threshold_values[:12]):  # Display only first 12 for Gradio
        _, binary_image = cv2.threshold(image_gray, thresh_val, 255, cv2.THRESH_BINARY)
        binary_image = morphology.remove_small_objects(binary_image > 0, min_size=20)

        # Label connected components
        label_image, num_labels = ndi.label(binary_image)

        # Display the thresholded image
        axes[i].imshow(binary_image, cmap='gray')
        axes[i].set_title(f'Threshold: {thresh_val:.0f}\nCount: {num_labels}')
        axes[i].axis('off')

    plt.tight_layout()
    
    # Return the processed plot
    return fig

# Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Plot(),
    title="Lipid Lens",
    description="Upload an image of lipid droplets to analyze them."
)

demo.launch()
