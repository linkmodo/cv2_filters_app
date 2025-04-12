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

def apply_threshold(image, threshold_value=127, max_value=255, threshold_type="binary"):
    """Apply manual threshold to the image.
    
    Args:
        image: Input image
        threshold_value: Threshold value (0-255)
        max_value: Maximum value for binary thresholding (0-255)
        threshold_type: Type of thresholding - binary, binary_inv, trunc, tozero, tozero_inv
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Map threshold type to OpenCV constants
    threshold_types = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV
    }
    
    # Apply threshold
    _, thresholded = cv2.threshold(gray, threshold_value, max_value, threshold_types[threshold_type])
    
    # Convert back to BGR if the input was BGR
    if len(image.shape) == 3:
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    
    return thresholded

def apply_adaptive_threshold(image, max_value=255, adaptive_method="mean", threshold_type="binary", 
                            block_size=11, constant=2):
    """Apply adaptive threshold to the image.
    
    Args:
        image: Input image
        max_value: Maximum value for binary thresholding (0-255)
        adaptive_method: Adaptive method - "mean" or "gaussian"
        threshold_type: Type of thresholding - "binary" or "binary_inv"
        block_size: Size of the pixel neighborhood (must be odd)
        constant: Constant subtracted from the mean or weighted mean
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Ensure block size is odd
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    # Map adaptive method to OpenCV constants
    adaptive_methods = {
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }
    
    # Map threshold type to OpenCV constants
    threshold_types = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV
    }
    
    # Apply adaptive threshold
    thresholded = cv2.adaptiveThreshold(
        gray, max_value, 
        adaptive_methods[adaptive_method],
        threshold_types[threshold_type],
        block_size, constant
    )
    
    # Convert back to BGR if the input was BGR
    if len(image.shape) == 3:
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    
    return thresholded

def detect_cameras(max_cameras=10):
    """
    Detect available camera devices.
    
    Args:
        max_cameras: Maximum number of camera indices to check
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

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
    elif operation == "Threshold":
        return apply_threshold(
            image,
            threshold_value=params.get("threshold_value", 127),
            max_value=params.get("max_value", 255),
            threshold_type=params.get("threshold_type", "binary")
        )
    elif operation == "Adaptive Threshold":
        return apply_adaptive_threshold(
            image,
            max_value=params.get("max_value", 255),
            adaptive_method=params.get("adaptive_method", "mean"),
            threshold_type=params.get("threshold_type", "binary"),
            block_size=params.get("block_size", 11),
            constant=params.get("constant", 2)
        )
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
        .filter-container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Input source selection
    input_source = st.radio("Select Input Source", ["File Upload", "Webcam"])
    
    if input_source == "File Upload":
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
                    
                    # Filter cascade controls
                    st.sidebar.subheader("Filter Cascade")
                    num_filters = st.sidebar.number_input("Number of Filters", 1, 5, 1)
                    
                    # Initialize filter parameters
                    filter_params = []
                    processed_image = image.copy()
                    
                    # Create filter containers
                    for i in range(num_filters):
                        with st.sidebar.expander(f"Filter {i+1}", expanded=True):
                            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                            
                            # Operation selection
                            operation = st.selectbox(
                                f"Operation {i+1}",
                                ["None", "Edge Detection (Sobel)", "Edge Detection (Canny)", "Sharpen", 
                                 "Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter",
                                 "Threshold", "Adaptive Threshold"],
                                key=f"operation_{i}"
                            )
                            
                            params = {}
                            
                            if operation != "None":
                                if operation == "Edge Detection (Sobel)" or operation == "Edge Detection (Canny)":
                                    st.subheader("Pre-processing Options")
                                    pre_blur = st.slider(
                                        "Pre-processing Blur Kernel Size",
                                        0, 21, 0, step=2,
                                        key=f"pre_blur_{i}"
                                    )
                                    if pre_blur > 0:
                                        blur_type = st.selectbox(
                                            "Blur Type",
                                            ["gaussian", "box"],
                                            key=f"blur_type_{i}"
                                        )
                                        params["blur_type"] = blur_type
                                        
                                        if blur_type == "gaussian":
                                            sigma = st.slider(
                                                "Gaussian Sigma",
                                                0.1, 5.0, 1.0,
                                                key=f"sigma_{i}"
                                            )
                                            params["sigma"] = sigma
                                    
                                    params["pre_blur"] = pre_blur
                                    
                                    st.subheader("Edge Detection Options")
                                    if operation == "Edge Detection (Sobel)":
                                        params["direction"] = st.selectbox(
                                            "Edge Direction",
                                            ["both", "x", "y"],
                                            key=f"direction_{i}"
                                        )
                                        params["threshold"] = st.slider(
                                            "Edge Threshold",
                                            0, 255, 30,
                                            key=f"threshold_{i}"
                                        )
                                    else:  # Canny
                                        params["threshold1"] = st.slider(
                                            "Lower Threshold",
                                            0, 255, 100,
                                            key=f"threshold1_{i}"
                                        )
                                        params["threshold2"] = st.slider(
                                            "Upper Threshold",
                                            0, 255, 200,
                                            key=f"threshold2_{i}"
                                        )
                                    
                                    params["enhance"] = st.checkbox(
                                        "Enhance Edges",
                                        value=True,
                                        key=f"enhance_{i}"
                                    )
                                    if params["enhance"]:
                                        color_option = st.selectbox(
                                            "Edge Color",
                                            ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"],
                                            key=f"color_{i}"
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
                                        params["line_thickness"] = st.slider(
                                            "Line Thickness",
                                            1, 5, 1,
                                            key=f"thickness_{i}"
                                        )
                                
                                elif operation == "Sharpen":
                                    params["intensity"] = st.selectbox(
                                        "Sharpening Intensity",
                                        ["normal", "intense"],
                                        key=f"intensity_{i}"
                                    )
                                
                                elif operation == "Box Blur":
                                    params["kernel_size"] = st.slider(
                                        "Kernel Size",
                                        3, 31, 5, step=2,
                                        key=f"box_kernel_{i}"
                                    )
                                
                                elif operation == "Gaussian Blur":
                                    params["kernel_size"] = st.slider(
                                        "Kernel Size",
                                        3, 31, 5, step=2,
                                        key=f"gauss_kernel_{i}"
                                    )
                                    params["sigma"] = st.slider(
                                        "Sigma",
                                        0, 10, 0,
                                        key=f"gauss_sigma_{i}"
                                    )
                                
                                elif operation == "Bilateral Filter":
                                    params["d"] = st.slider(
                                        "Diameter",
                                        1, 15, 9, step=2,
                                        key=f"bilateral_d_{i}"
                                    )
                                    params["sigma_color"] = st.slider(
                                        "Color Sigma",
                                        1, 100, 75, step=1,
                                        key=f"bilateral_color_{i}"
                                    )
                                    params["sigma_space"] = st.slider(
                                        "Space Sigma",
                                        1, 100, 75, step=1,
                                        key=f"bilateral_space_{i}"
                                    )
                                
                                elif operation == "Median Filter":
                                    params["kernel_size"] = st.slider(
                                        "Kernel Size",
                                        3, 51, 5, step=2,
                                        key=f"median_kernel_{i}"
                                    )
                                elif operation == "Threshold":
                                    params["threshold_value"] = st.slider(
                                        "Threshold Value",
                                        0, 255, 127, step=1,
                                        key=f"threshold_value_{i}"
                                    )
                                    params["max_value"] = st.slider(
                                        "Max Value",
                                        0, 255, 255, step=1,
                                        key=f"max_value_{i}"
                                    )
                                    params["threshold_type"] = st.selectbox(
                                        "Threshold Type",
                                        ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"],
                                        key=f"threshold_type_{i}"
                                    )
                                elif operation == "Adaptive Threshold":
                                    params["max_value"] = st.slider(
                                        "Max Value",
                                        0, 255, 255, step=1,
                                        key=f"adaptive_max_value_{i}"
                                    )
                                    params["adaptive_method"] = st.selectbox(
                                        "Adaptive Method",
                                        ["mean", "gaussian"],
                                        key=f"adaptive_method_{i}"
                                    )
                                    params["threshold_type"] = st.selectbox(
                                        "Threshold Type",
                                        ["binary", "binary_inv"],
                                        key=f"adaptive_threshold_type_{i}"
                                    )
                                    params["block_size"] = st.slider(
                                        "Block Size",
                                        3, 99, 11, step=2,
                                        key=f"block_size_{i}"
                                    )
                                    params["constant"] = st.slider(
                                        "Constant",
                                        0, 20, 2, step=1,
                                        key=f"constant_{i}"
                                    )
                            
                            filter_params.append((operation, params))
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.sidebar.button("Apply Filters"):
                        try:
                            # Apply image adjustments first
                            processed_image = adjust_image(
                                processed_image,
                                brightness=brightness,
                                contrast=contrast,
                                saturation=saturation
                            )
                            
                            # Apply filters in sequence
                            for operation, params in filter_params:
                                if operation != "None":
                                    processed_image = process_image(processed_image, operation, params)
                            
                            # Display results
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
                        ["Edge Detection (Sobel)", "Edge Detection (Canny)", "Sharpen", 
                         "Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter",
                         "Threshold", "Adaptive Threshold"]
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
                                3, 51, 5, step=2
                            )
                            params["kernel_size"] = kernel_size
                    elif operation == "Threshold":
                        params["threshold_value"] = st.sidebar.slider(
                            "Threshold Value",
                            0, 255, 127, step=1
                        )
                        params["max_value"] = st.sidebar.slider(
                            "Max Value",
                            0, 255, 255, step=1
                        )
                        params["threshold_type"] = st.sidebar.selectbox(
                            "Threshold Type",
                            ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"]
                        )
                    elif operation == "Adaptive Threshold":
                        params["max_value"] = st.sidebar.slider(
                            "Max Value",
                            0, 255, 255, step=1
                        )
                        params["adaptive_method"] = st.sidebar.selectbox(
                            "Adaptive Method",
                            ["mean", "gaussian"]
                        )
                        params["threshold_type"] = st.sidebar.selectbox(
                            "Threshold Type",
                            ["binary", "binary_inv"]
                        )
                        params["block_size"] = st.sidebar.slider(
                            "Block Size",
                            3, 99, 11, step=2
                        )
                        params["constant"] = st.sidebar.slider(
                            "Constant",
                            0, 20, 2, step=1
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
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Please make sure you've uploaded a valid image or video file.")

    else:  # Webcam
        st.header("Webcam Processing")
        
        # Camera detection
        webcam_col1, webcam_col2 = st.columns([3, 1])
        
        with webcam_col2:
            if st.button("Scan for Cameras"):
                with st.spinner("Scanning for available cameras..."):
                    available_cameras = detect_cameras()
                    if available_cameras:
                        st.success(f"Found {len(available_cameras)} camera(s): {available_cameras}")
                    else:
                        st.error("No cameras detected. Please check your connections.")
        
        with webcam_col1:
            st.info("If your webcam isn't working, try clicking 'Scan for Cameras' to find available camera indices, or try installing the 'opencv-python-headless' package.")
        
        # Sidebar controls for webcam
        st.sidebar.header("Processing Options")
        operation = st.sidebar.selectbox(
            "Select Operation",
            ["None", "Edge Detection (Sobel)", "Edge Detection (Canny)", "Sharpen", 
             "Box Blur", "Gaussian Blur", "Bilateral Filter", "Median Filter",
             "Threshold", "Adaptive Threshold"]
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
        elif operation == "Box Blur":
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
                3, 51, 5, step=2
            )
            params["kernel_size"] = kernel_size
        elif operation == "Threshold":
            params["threshold_value"] = st.sidebar.slider(
                "Threshold Value",
                0, 255, 127, step=1
            )
            params["max_value"] = st.sidebar.slider(
                "Max Value",
                0, 255, 255, step=1
            )
            params["threshold_type"] = st.sidebar.selectbox(
                "Threshold Type",
                ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"]
            )
        elif operation == "Adaptive Threshold":
            params["max_value"] = st.sidebar.slider(
                "Max Value",
                0, 255, 255, step=1
            )
            params["adaptive_method"] = st.sidebar.selectbox(
                "Adaptive Method",
                ["mean", "gaussian"]
            )
            params["threshold_type"] = st.sidebar.selectbox(
                "Threshold Type",
                ["binary", "binary_inv"]
            )
            params["block_size"] = st.sidebar.slider(
                "Block Size",
                3, 99, 11, step=2
            )
            params["constant"] = st.sidebar.slider(
                "Constant",
                0, 20, 2, step=1
            )
        
        # Add option to select webcam source
        camera_index = st.sidebar.selectbox(
            "Camera Source", 
            options=[0, 1, 2, 3, "Custom"],
            help="Select webcam device index. Try different numbers if default (0) doesn't work."
        )
        
        if camera_index == "Custom":
            camera_index = st.sidebar.number_input("Enter custom camera index", min_value=0, value=0)
        
        # Display webcam
        webcam_placeholder = st.empty()
        start_button = st.button("Start Processing")
        stop_button = st.button("Stop Processing")
        
        if start_button:
            try:
                # Display attempting to connect message
                connecting_message = st.info(f"Attempting to connect to camera index {camera_index}...")
                
                # Try different camera APIs (useful for Windows especially)
                # First try default API
                cap = cv2.VideoCapture(camera_index)
                
                # If that fails, try DirectShow API (on Windows)
                if not cap.isOpened():
                    st.info("First attempt failed. Trying DirectShow API...")
                    # DirectShow (DSHOW) API - often works better on Windows
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                
                # If still fails, try using Media Foundation API (on newer Windows)
                if not cap.isOpened():
                    st.info("Second attempt failed. Trying Media Foundation API...")
                    # Media Foundation API - for Windows 8 and newer
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
                
                # Check if any of the attempts successfully opened the camera
                if not cap.isOpened():
                    st.error(f"Could not open webcam at index {camera_index}. Please try a different index or check your camera connection.")
                    
                    # Add more detailed troubleshooting suggestions
                    st.warning("""
                    Troubleshooting tips:
                    1. Try selecting a different camera index from the dropdown
                    2. Check if your webcam is connected properly
                    3. Make sure no other application is using your webcam
                    4. For Windows users, check if privacy settings allow browser access to camera
                    5. Try running the app with 'pip install opencv-python' instead of 'opencv-python-headless'
                    6. Try restarting your computer
                    """)
                    
                    # Suggest alternatives
                    st.info("Alternative: If you're unable to use your webcam directly, consider taking a photo with your device and uploading it through the 'File Upload' option.")
                else:
                    # Clear connecting message
                    connecting_message.empty()
                    st.success(f"Successfully connected to camera at index {camera_index}")
                    
                    is_running = True
                    while is_running and not stop_button:
                        # Read frame
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from webcam.")
                            break
                        
                        # Process frame if an operation is selected
                        if operation != "None":
                            processed_frame = process_image(frame, operation, params)
                        else:
                            processed_frame = frame
                        
                        # Convert to RGB for display
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the frame
                        webcam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                    # Release webcam when stopped
                    cap.release()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check if your webcam is properly connected and not in use by another application.")

    # Add footer at the end
    st.markdown(
        '<div class="footer">Built by Li Fan • Powered by Streamlit & CV2</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
