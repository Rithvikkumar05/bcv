import streamlit as st
import cv2
from PIL import Image
import numpy as np

def canny_filter(image, min_val, max_val):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, min_val, max_val)
    return edges

def log_filter(image):
    # Applying the Log filter to the image
    log_transformed = np.log1p(image.astype(np.float32))
    normalized = cv2.normalize(log_transformed, None, 0, 255, cv2.NORM_MINMAX)
    log_filtered = np.uint8(normalized)
    return log_filtered

def dog_filter(image, ksize=3, sigma=1.0):
    # Applying the Difference of Gaussians (DoG) filter to the image
    blurred1 = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    blurred2 = cv2.GaussianBlur(image, (ksize * 2 + 1, ksize * 2 + 1), sigma * 2)

    dog_filtered = cv2.subtract(blurred1, blurred2)
    return dog_filtered

def apply_filter(image, filter_type, min_val, max_val):
    if filter_type == 'Canny':
        return canny_filter(image, min_val, max_val)
    elif filter_type == 'Log':
        return log_filter(image)
    elif filter_type == 'Dog':
        return dog_filter(image)
    else:
        st.warning("Invalid filter type selected.")
        return None

def main():
    st.title('Image Filter App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)

        filter_type = st.selectbox("Select Filter Type", ['Canny', 'Log', 'Dog'])
        min_threshold = st.slider('Min Threshold:', 0, 255, 50)
        max_threshold = st.slider('Max Threshold:', 0, 255, 150)

        if st.button('Apply Filter'):
            img_array = np.array(image)
            filtered_image = apply_filter(img_array, filter_type, min_threshold, max_threshold)
            if filtered_image is not None:
                st.image(filtered_image, caption=f"{filter_type} Filter Result", use_column_width=True)

if __name__ == "__main__":
    main()
