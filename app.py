import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

def apply_box_blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_gaussian_blur(image, kernel_size, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_sharpen(image, intensity='normal'):
    if intensity == 'normal':
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
    else:  # intense
        kernel = np.array([[0, -4, 0],
                          [-4, 17, -4],
                          [0, -4, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_sobel_edge(image, direction='both', threshold=30, enhance=False, line_color=(0, 0, 255), line_thickness=1):
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Calculate gradients
    if direction in ['x', 'both']:
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    if direction in ['y', 'both']:
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradients based on direction
    if direction == 'x':
        gradient = np.absolute(sobelx)
    elif direction == 'y':
        gradient = np.absolute(sobely)
    else:  # both
        gradient = np.sqrt(sobelx**2 + sobely**2)

    # Normalize and convert to uint8
    gradient = np.uint8(gradient * 255 / gradient.max())

    # Apply threshold
    _, binary_edges = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)

    if enhance:
        # Create a 3-channel image for colored output
        enhanced = image.copy()
        if len(enhanced.shape) == 2:  # If input was grayscale
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Create a mask for the edges with specified thickness
        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            binary_edges = cv2.dilate(binary_edges, kernel, iterations=1)
        
        # Overlay edges with specified color
        enhanced[binary_edges > 0] = line_color
        return enhanced
    else:
        # Return binary edges if no enhancement requested
        if len(image.shape) == 3:  # If input was color
            binary_edges = cv2.cvtColor(binary_edges, cv2.COLOR_GRAY2BGR)
        return binary_edges

def process_image(image, operation, params):
    if operation == "Edge Detection":
        return apply_sobel_edge(
            image,
            direction=params.get("direction", "both"),
            threshold=params.get("threshold", 30),
            enhance=params.get("enhance", False),
            line_color=params.get("line_color", (0, 0, 255)),
            line_thickness=params.get("line_thickness", 1)
        )
    elif operation == "Sharpen":
        return apply_sharpen(image, params.get("intensity", "normal"))
    elif operation == "Box Blur":
        return apply_box_blur(image, params["kernel_size"])
    elif operation == "Gaussian Blur":
        return apply_gaussian_blur(image, params["kernel_size"], params.get("sigma", 0))
    return image

def main():
    st.title("Image Processing Tool")
    
    # Add custom CSS for footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: transparent;
            color: grey;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])
    
    if uploaded_file is not None:
        # Determine if it's an image or video
        file_type = uploaded_file.type
        
        if file_type.startswith('image'):
            # Handle image processing
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            # Convert BGR to RGB for display
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            st.image(image, caption="Original Image", channels="BGR")
            
            # Sidebar controls
            st.sidebar.header("Processing Options")
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Edge Detection", "Sharpen", "Box Blur", "Gaussian Blur"]
            )
            
            params = {}
            if operation == "Edge Detection":
                params["direction"] = st.sidebar.selectbox(
                    "Edge Direction",
                    ["both", "x", "y"]
                )
                params["threshold"] = st.sidebar.slider(
                    "Edge Threshold",
                    0, 255, 30
                )
                params["enhance"] = st.sidebar.checkbox(
                    "Enhance Edges",
                    value=True,
                    help="Overlay detected edges on the original image"
                )
                if params["enhance"]:
                    color_option = st.sidebar.selectbox(
                        "Edge Color",
                        ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
                    )
                    # Map color names to BGR values
                    color_map = {
                        "Red": (0, 0, 255),
                        "Green": (0, 255, 0),
                        "Blue": (255, 0, 0),
                        "Yellow": (0, 255, 255),
                        "Cyan": (255, 255, 0),
                        "Magenta": (255, 0, 255),
                        "White": (255, 255, 255)
                    }
                    params["line_color"] = color_map[color_option]
                    params["line_thickness"] = st.sidebar.slider(
                        "Line Thickness",
                        1, 5, 1
                    )
            elif operation == "Sharpen":
                params["intensity"] = st.sidebar.selectbox(
                    "Sharpening Intensity",
                    ["normal", "intense"]
                )
            elif operation in ["Box Blur", "Gaussian Blur"]:
                params["kernel_size"] = st.sidebar.slider(
                    "Kernel Size",
                    3, 31, 5, step=2
                )
                if operation == "Gaussian Blur":
                    params["sigma"] = st.sidebar.slider(
                        "Sigma",
                        0, 10, 0
                    )
            
            if st.sidebar.button("Apply Filter"):
                processed_image = process_image(image, operation, params)
                st.image(processed_image, caption="Processed Image", channels="BGR")
                
                # Create download button for processed image
                processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                processed_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
                
        elif file_type.startswith('video'):
            # Handle video processing
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.sidebar.header("Processing Options")
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Edge Detection", "Sharpen", "Box Blur", "Gaussian Blur"]
            )
            
            params = {}
            if operation == "Edge Detection":
                params["direction"] = st.sidebar.selectbox(
                    "Edge Direction",
                    ["both", "x", "y"]
                )
                params["threshold"] = st.sidebar.slider(
                    "Edge Threshold",
                    0, 255, 30
                )
                params["enhance"] = st.sidebar.checkbox(
                    "Enhance Edges",
                    value=True,
                    help="Overlay detected edges on the original image"
                )
                if params["enhance"]:
                    color_option = st.sidebar.selectbox(
                        "Edge Color",
                        ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
                    )
                    # Map color names to BGR values
                    color_map = {
                        "Red": (0, 0, 255),
                        "Green": (0, 255, 0),
                        "Blue": (255, 0, 0),
                        "Yellow": (0, 255, 255),
                        "Cyan": (255, 255, 0),
                        "Magenta": (255, 0, 255),
                        "White": (255, 255, 255)
                    }
                    params["line_color"] = color_map[color_option]
                    params["line_thickness"] = st.sidebar.slider(
                        "Line Thickness",
                        1, 5, 1
                    )
            elif operation == "Sharpen":
                params["intensity"] = st.sidebar.selectbox(
                    "Sharpening Intensity",
                    ["normal", "intense"]
                )
            elif operation in ["Box Blur", "Gaussian Blur"]:
                params["kernel_size"] = st.sidebar.slider(
                    "Kernel Size",
                    3, 31, 5, step=2
                )
                if operation == "Gaussian Blur":
                    params["sigma"] = st.sidebar.slider(
                        "Sigma",
                        0, 10, 0
                    )
            
            if st.sidebar.button("Process Video"):
                video = cv2.VideoCapture(tfile.name)
                
                # Get video properties
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                
                # Create temporary file for processed video
                processed_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(
                    processed_video_file.name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, (width, height)
                )
                
                # Process video
                progress_bar = st.progress(0)
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                
                for i in range(frame_count):
                    ret, frame = video.read()
                    if ret:
                        processed_frame = process_image(frame, operation, params)
                        out.write(processed_frame)
                        progress_bar.progress((i + 1) / frame_count)
                
                video.release()
                out.release()
                
                # Provide download link for processed video
                with open(processed_video_file.name, 'rb') as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

    # Add footer at the end
    st.markdown(
        '<div class="footer">Built by Li Fan • Powered by Streamlit & CV2</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
