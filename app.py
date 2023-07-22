# Python In-built packages
from pathlib import Path
import PIL
import cv2
# External packages
import streamlit as st
import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os
from PIL import Image
import numpy as np

# Local Modules
import settings

# Setting page layout
st.set_page_config(
    page_title="Brain Tumor Detection and Segmentation using Detectron2",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Brain Tumor Detection and Segmentation using Detectron2")

# Sidebar
st.sidebar.header("Detectron2 Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100


# Load Pre-trained ML Model
try:
    # load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = './output/model_final.pth'
    cfg.MODEL.DEVICE = 'cpu'

    predictor = DefaultPredictor(cfg)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: model_path")
    st.error(ex)

source_img = None
# If image is selected
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                        use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img).convert('RGB')
            st.image(source_img, caption="Uploaded Image",
                        use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                outputs = predictor(uploaded_image)
                threshold = confidence
                
                # Display predictions
                preds = outputs["instances"].pred_classes.tolist()
                scores = outputs["instances"].scores.tolist()
                bboxes = outputs["instances"].pred_boxes
                
                v = Visualizer(uploaded_image[:, :, ::-1], metadata = MetadataCatalog.get("brain_tumor_test"), scale=0.7, instance_mode=ColorMode.IMAGE_BW)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
                
                res_plotted = img[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        bboxes_ = []
                        for j, bbox in enumerate(bboxes):
                            bbox = bbox.tolist()

                            score = scores[j]
                            pred = preds[j]
                            

                            if score > threshold:
                                x1, y1, x2, y2 = [int(i) for i in bbox]
                                bboxes_.append([x1, y1, x2, y2])
                            st.write(pred)    
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

