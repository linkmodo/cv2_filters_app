import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

def adjust_image(image, brightness=0, contrast=1.0, saturation=1.0):
    # Convert to float for calculations
    adjusted = image.astype(float)
    
    # Adjust brightness (add/subtract)
    adjusted = adjusted + brightness
    
    # Adjust contrast (multiply)
    adjusted = adjusted * contrast
    
    # Convert to HSV for saturation adjustment
    if len(image.shape) == 3:  # Color image
        hsv = cv2.cvtColor(np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)  # Adjust S channel
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Ensure values are in valid range
    return np.clip(adjusted, 0, 255).astype(np.uint8)

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

def apply_preprocessing_blur(image, blur_type='gaussian', kernel_size=5, sigma=0):
    """
    Apply pre-processing blur to an image
    Args:
        image: Input image
        blur_type: 'gaussian' or 'box'
        kernel_size: Size of the blur kernel (must be odd)
        sigma: Standard deviation for Gaussian blur
    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if blur_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    else:  # box blur
        return cv2.blur(image, (kernel_size, kernel_size))

def apply_canny_edge(image, threshold1=100, threshold2=200, enhance=False, line_color=(0, 0, 255), 
                    line_thickness=1, pre_blur=0, blur_type='gaussian', sigma=0):
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply pre-processing blur if specified
    if pre_blur > 0:
        gray = apply_preprocessing_blur(gray, blur_type, pre_blur, sigma)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    if enhance:
        # Create a 3-channel image for colored output
        enhanced = image.copy()
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Create a mask for the edges with specified thickness
        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Overlay edges with specified color
        enhanced[edges > 0] = line_color
        return enhanced
    else:
        if len(image.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

def apply_sobel_edge(image, direction='both', threshold=30, enhance=False, line_color=(0, 0, 255), 
                    line_thickness=1, pre_blur=0, blur_type='gaussian', sigma=0):
    try:
        # Convert to grayscale if image is color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply pre-processing blur if specified
        if pre_blur > 0:
            gray = apply_preprocessing_blur(gray, blur_type, pre_blur, sigma)

        # Calculate gradients
        if direction in ['x', 'both']:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        if direction in ['y', 'both']:
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

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
            if len(enhanced.shape) == 2:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Create a mask for the edges with specified thickness
            if line_thickness > 1:
                kernel = np.ones((line_thickness, line_thickness), np.uint8)
                binary_edges = cv2.dilate(binary_edges, kernel, iterations=1)
            
            # Overlay edges with specified color
            enhanced[binary_edges > 0] = line_color
            return enhanced
        else:
            if len(image.shape) == 3:
                binary_edges = cv2.cvtColor(binary_edges, cv2.COLOR_GRAY2BGR)
            return binary_edges
    except Exception as e:
        st.error(f"Error in edge detection: {str(e)}")
        return image

def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    """Apply bilateral filter to the image.
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_median_filter(image, kernel_size):
    """Apply median filter to the image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
    """
    return cv2.medianBlur(image, kernel_size)

def process_image(image, operation, params):
    # Apply image adjustments first if they exist
    if "brightness" in params or "contrast" in params or "saturation" in params:
        image = adjust_image(
            image,
            brightness=params.get("brightness", 0),
            contrast=params.get("contrast", 1.0),
            saturation=params.get("saturation", 1.0)
        )
    
    if operation == "Edge Detection (Sobel)":
        return apply_sobel_edge(
            image,
            direction=params.get("direction", "both"),
            threshold=params.get("threshold", 30),
            enhance=params.get("enhance", False),
            line_color=params.get("line_color", (0, 0, 255)),
            line_thickness=params.get("line_thickness", 1),
            pre_blur=params.get("pre_blur", 0),
            blur_type=params.get("blur_type", "gaussian"),
            sigma=params.get("sigma", 0)
        )
    elif operation == "Edge Detection (Canny)":
        return apply_canny_edge(
            image,
            threshold1=params.get("threshold1", 100),
            threshold2=params.get("threshold2", 200),
            enhance=params.get("enhance", False),
            line_color=params.get("line_color", (0, 0, 255)),
            line_thickness=params.get("line_thickness", 1),
            pre_blur=params.get("pre_blur", 0),
            blur_type=params.get("blur_type", "gaussian"),
            sigma=params.get("sigma", 0)
        )
    elif operation == "Sharpen":
        return apply_sharpen(image, params.get("intensity", "normal"))
    elif operation == "Box Blur":
        return apply_box_blur(image, params["kernel_size"])
    elif operation == "Gaussian Blur":
        return apply_gaussian_blur(image, params["kernel_size"], params.get("sigma", 0))
    elif operation == "Bilateral Filter":
        d = params["d"]
        sigma_color = params["sigma_color"]
        sigma_space = params["sigma_space"]
        return apply_bilateral_filter(image, d, sigma_color, sigma_space)
    elif operation == "Median Filter":
        kernel_size = params["kernel_size"]
        return apply_median_filter(image, kernel_size)
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
        try:
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
                
                # Image adjustment controls
                st.sidebar.subheader("Image Adjustments")
                brightness = st.sidebar.slider("Brightness", -100, 100, 0)
                contrast = st.sidebar.slider("Contrast", 0.0, 3.0, 1.0)
                saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0)
                
                # Operation selection
                st.sidebar.subheader("Filter Selection")
                operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Edge Detection (Sobel)", "Edge Detection (Canny)", "Sharpen", "Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter"]
                )
                
                params = {
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation
                }
                
                if operation == "Edge Detection (Sobel)" or operation == "Edge Detection (Canny)":
                    st.sidebar.subheader("Pre-processing Options")
                    # Add pre-blur option
                    pre_blur = st.sidebar.slider(
                        "Pre-processing Blur Kernel Size",
                        0, 21, 0, step=2,
                        help="Apply blur before edge detection (0 for no blur)"
                    )
                    if pre_blur > 0:
                        blur_type = st.sidebar.selectbox(
                            "Blur Type",
                            ["gaussian", "box"],
                            help="Choose the type of blur to apply"
                        )
                        params["blur_type"] = blur_type
                        
                        if blur_type == "gaussian":
                            sigma = st.sidebar.slider(
                                "Gaussian Sigma",
                                0.1, 5.0, 1.0, 
                                help="Standard deviation for Gaussian blur"
                            )
                            params["sigma"] = sigma
                        
                        st.sidebar.info(f"Using {pre_blur}x{pre_blur} {blur_type} blur kernel")
                    
                    params["pre_blur"] = pre_blur
                    
                    st.sidebar.subheader("Edge Detection Options")
                    if operation == "Edge Detection (Sobel)":
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
                    else:  # Canny
                        params["threshold1"] = st.sidebar.slider(
                            "Lower Threshold",
                            0, 255, 100
                        )
                        params["threshold2"] = st.sidebar.slider(
                            "Upper Threshold",
                            0, 255, 200
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
                elif operation in ["Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter"]:
                    if operation == "Box Blur":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        params["kernel_size"] = kernel_size
                    elif operation == "Gaussian Blur":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        sigma = st.sidebar.slider(
                            "Sigma",
                            0, 10, 0
                        )
                        params["kernel_size"] = kernel_size
                        params["sigma"] = sigma
                    elif operation == "Bilateral Filter":
                        d = st.sidebar.slider(
                            "Diameter",
                            1, 15, 9, step=2
                        )
                        sigma_color = st.sidebar.slider(
                            "Color Sigma",
                            1, 100, 75, step=1
                        )
                        sigma_space = st.sidebar.slider(
                            "Space Sigma",
                            1, 100, 75, step=1
                        )
                        params["d"] = d
                        params["sigma_color"] = sigma_color
                        params["sigma_space"] = sigma_space
                    elif operation == "Median Filter":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        params["kernel_size"] = kernel_size
                
                if st.sidebar.button("Apply Filter"):
                    try:
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
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                
            elif file_type.startswith('video'):
                # Handle video processing
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                st.sidebar.header("Processing Options")
                operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Edge Detection (Sobel)", "Edge Detection (Canny)", "Sharpen", "Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter"]
                )
                
                params = {}
                if operation == "Edge Detection (Sobel)":
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
                elif operation == "Edge Detection (Canny)":
                    params["threshold1"] = st.sidebar.slider(
                        "Lower Threshold",
                        0, 255, 100
                    )
                    params["threshold2"] = st.sidebar.slider(
                        "Upper Threshold",
                        0, 255, 200
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
                elif operation in ["Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter"]:
                    if operation == "Box Blur":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        params["kernel_size"] = kernel_size
                    elif operation == "Gaussian Blur":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        sigma = st.sidebar.slider(
                            "Sigma",
                            0, 10, 0
                        )
                        params["kernel_size"] = kernel_size
                        params["sigma"] = sigma
                    elif operation == "Bilateral Filter":
                        d = st.sidebar.slider(
                            "Diameter",
                            1, 15, 9, step=2
                        )
                        sigma_color = st.sidebar.slider(
                            "Color Sigma",
                            1, 100, 75, step=1
                        )
                        sigma_space = st.sidebar.slider(
                            "Space Sigma",
                            1, 100, 75, step=1
                        )
                        params["d"] = d
                        params["sigma_color"] = sigma_color
                        params["sigma_space"] = sigma_space
                    elif operation == "Median Filter":
                        kernel_size = st.sidebar.slider(
                            "Kernel Size",
                            3, 31, 5, step=2
                        )
                        params["kernel_size"] = kernel_size
                
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
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please make sure you've uploaded a valid image or video file.")

    # Add footer at the end
    st.markdown(
        '<div class="footer">Built by Li Fan • Powered by Streamlit & CV2</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
