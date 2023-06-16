import cv2
from transformers import AutoImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Load the image processor and depth estimation model
image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")#, cache_dir = "./ImageProcessor")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")#, cache_dir = "./DepthEstimationModel")

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index if multiple cameras are connected

while cap.isOpened():#True:
    # Capture frame from the camera
    ret, frame = cap.read()

    # Convert the frame to PIL Image format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert depth prediction to numpy array and visualize
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    # Display the depth estimation and original frame
    cv2.imshow("Depth Estimation", np.array(depth))
    cv2.imshow("Original Frame", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
