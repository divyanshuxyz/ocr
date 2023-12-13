import cv2
import streamlit as st
import easyocr
import numpy as np
# Load the OCR model
reader = easyocr.Reader(['en'], gpu=False)

# Streamlit app
def main():
    st.title("Optical Character Recognition")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Perform OCR
        text_results = reader.readtext(image)
        text = [result[1] for result in text_results]

        # Display image with bounding boxes
        for (bbox, text, prob) in text_results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw the bounding box
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # Display the text
            cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the image with bounding boxes
        st.image(image, channels="BGR", caption="Uploaded Image with Bounding Boxes", use_column_width=True)

        # Display recognized text
        st.subheader("Recognized Text:")
        text = [result[1] for result in text_results]
        st.write(text)

if __name__ == "__main__":
    main()
