# Image and Video Edge Detection & Filter Cascade Tool

This project is a **Streamlit-based image processing tool** developed for personal experimentation with **OCT layer segmentation**, but it has evolved into a versatile app that handles **edge detection** and **filter cascades** for both images and videos.

Initially focused on OCT image processing, the application now includes support for:
- Customizable **Sobel and Canny edge detection**
- A suite of **blurring, sharpening, and color adjustment filters**
- A **filter cascade pipeline** that allows multiple transformations in sequence
- Upload support for **image and video** files
- Downloadable output after processing

### 🌟 Highlights
- **Cascading Filter Pipeline**: Apply multiple filters in a user-defined order.
- **Edge Enhancement**: Sobel and Canny with color overlay and line thickness.
- **Median Filter Integration**: Recently added to improve noise reduction before edge detection.
- **Real-Time Adjustment**: Interactive sliders for tuning brightness, contrast, saturation, and kernel sizes.
- **Video Processing**: Frame-by-frame transformation with download support.

### 📸 Supported Filters & Effects
- Sobel Edge Detection (x, y, both)
- Canny Edge Detection
- Median Filter
- Gaussian Blur
- Box Blur
- Bilateral Filter
- Sharpen (normal & intense)
- Color & brightness adjustments

### ⚙️ How It Works
- Upload an image or video.
- Select one or more filters to apply.
- Fine-tune each filter's parameters interactively.
- View the result in real time.
- Download the processed image or video.

### 📦 Requirements
Make sure to install required packages:
```bash
pip install streamlit opencv-python-headless numpy pillow
```

### 🚀 Run the App
```bash
streamlit run app.py
```

### 📂 File Structure
```
.
├── app.py
├── README.md
```

### 📌 Notes
- This is still a **proof-of-concept**. The goal is to eventually build a **standalone Windows GUI** app with **drawing capabilities** on top of edge-detected lines.
- This version reflects an accumulation of iterations and class-learned techniques, particularly the **median filter** and **structured filter chaining**.

### 🙋‍♂️ Author
Built by **Li Fan** – March 2025  
_Powered by Streamlit & OpenCV_

