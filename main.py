import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model_path = os.path.join('model', 'ship-model.h5')

@st.cache(allow_output_mutation=True)
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model


def find_regions(image, method):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    if method == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()
    boxes = []
    for (x, y, w, h) in rects:
        boxes.append([x, y, w, h])
        pass

    return boxes


def scene_preprocess(scene):
    scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    return scene


def predict(model, scene):
    rois, locs = [], []
    box_in_scene = find_regions(scene, method="fast")
    (H, W) = scene.shape[:2]
    for (x, y, w, h) in box_in_scene:

        if w / float(W) > 0.10 and h / float(H) > 0.10:
            continue

        roi = scene[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (48, 48))

        rois.append(roi)
        locs.append((x, y, x + w, y + h))
        pass

    preds = model.predict(np.array(rois, dtype=np.float32))
    preds = np.argmax(preds, axis=1)

    img = scene.copy()
    for (i, label) in enumerate(preds):

        if label == 1:
            (startX, startY, endX, endY) = locs[i]
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return img


def cs_detect_ship():
    st.markdown("<h1 style='text-align: center; color: black;'>Detecting Ships in Satellite Imagery</h1>",
                unsafe_allow_html=True)

    Image_o = st.file_uploader('Upload image here', type=['jpg', 'jpeg', 'png'])
    my_expander = st.expander(label='🙋 Upload help')
    with my_expander:
        st.markdown('Filetype to upload : **JPG, JPEG, PNG**')
    if Image_o is not None:
        col1, col2 = st.columns([1, 2])
        new_image = Image_o.read()
        with col1:
            col1.subheader("Uploaded Scene")
            st.image(new_image)
            img = predict(model_load(), scene_preprocess(np.array(Image.open(BytesIO(new_image)))))
        with col2:
            col2.subheader("Detected Ships")
            st.image(img)

    return None


def main():
    st.set_page_config(
        page_title='Detecting Ships',
        layout="wide",
    )
    cs_detect_ship()
    return None


if __name__ == '__main__':
    main()
