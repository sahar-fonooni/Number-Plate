# !pip install ultralytics==8.2.100

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image




from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def fit_logo(base_image, logo, box):


  # Calculate the size of the box
  box_width = box[2] - box[0]
  box_height = box[3] - box[1]

  # Resize the image to fit
  logo = logo.resize((box_width, box_height), Image.LANCZOS)

  # Paste the resized image into the base image at the specified box location
  base_image.paste(logo, box)

  return base_image

def cover_plate(model, image, logo):
  results = model.predict(source=image, conf=0.25)
  result = int_list = list(map(int, results[0].boxes.xyxy.tolist()[0]))
  base_image = Image.open(image)
  covered_image = fit_logo(base_image, logo, result)
  return covered_image




# test the model
model = YOLO("/content/best (3).pt")
image = "/content/3.jpg"
logo = Image.open('/content/oto.jfif') # you just need to upload the logo file


covered_image = cover_plate(model, image, logo)
image_array = np.array(covered_image)

plt.imshow(image_array)