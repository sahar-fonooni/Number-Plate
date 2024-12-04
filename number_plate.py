import ultralytics
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def transform_directions(cordinates, rect_points, angle):
  dst_points = np.array([[0, 0], [0, min(cordinates) - 2], [max(cordinates) - 2, min(cordinates) - 2],\
                         [max(cordinates) - 2, 0]], dtype=np.float32)
  new_rect_points = rect_points.copy()
  if angle>=45 and angle<90:
    return dst_points, new_rect_points
  if angle>1 and angle<45:

    first = rect_points[:, :, 0].flatten()
    second = rect_points[:, :, 1].flatten()
    new_rect_points[0,0,:] = rect_points[0,second.argmin()].copy()
    new_rect_points[0,1,:] = rect_points[0,first.argmin(),:].copy()
    new_rect_points[0,2,:] = rect_points[0,second.argmax()].copy()
    new_rect_points[0,3,:] = rect_points[0,first.argmax(),:].copy()
    return dst_points, new_rect_points


  new_rect_points_1 = new_rect_points.copy()

  # sum x,y
  row_sums = np.sum(new_rect_points, axis=2)


  top_left_idx = row_sums.argmin()  # min top-left
  top_left = new_rect_points[0, top_left_idx].copy()      # بالا-چپ
  new_rect_points = np.delete(new_rect_points, top_left_idx, axis=1)

  row_sums = np.sum(new_rect_points, axis=2)
  bottom_right_idx = row_sums.argmax()  # max buttom-right
  bottom_right = new_rect_points[0, bottom_right_idx].copy()  # پایین-راست
  new_rect_points = np.delete(new_rect_points, bottom_right_idx, axis=1)




  if new_rect_points[0,0,0] < new_rect_points[0, 1, 0]:
      bottom_left = new_rect_points[0, 0].copy()   # پایین-چپ
      top_right = new_rect_points[0, 1].copy()     # بالا-راست
  else:
      bottom_left = new_rect_points[0, 1].copy()   # پایین-چپ
      top_right = new_rect_points[0, 0].copy()     # بالا-راست

  new_rect_points_1 = np.array([[top_left, bottom_left, bottom_right, top_right]])

  return dst_points ,new_rect_points_1



def cover_plate(model, image, logo):
  # Call model
  results = model.predict(source=image, conf=0.7, verbose=False)

  # Load the images
  base_image = cv2.imread(image)
  overlay_image = cv2.imread(logo)

  # Change Colour Scheme
  base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
  overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

  # Fetch the Detected Box
  rect_points = np.array(np.intp(results[0].obb.xyxyxyxy.tolist()))

  for i in rect_points:
    # rect_point = [i.tolist()]
    rect_pointt = np.array(i, dtype=np.int32)
    rect_point = np.expand_dims(rect_pointt, axis=0)

    # Measure the cordinates
    centers, cordinates, angle = cv2.minAreaRect(rect_point)

    # Define the destination point and transfor the directions
    dst_points, rect_point = transform_directions(cordinates, rect_point, angle)

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(dst_points, rect_point.astype(np.float32))

    # Resize the overlay image to the destination size
    cordinates = np.intp(cordinates)

    resized_overlay = cv2.resize(overlay_image, (max(cordinates),min(cordinates)), interpolation=cv2.INTER_AREA)

    # Warp the overlay image to fit the rotated rectangle using the perspective transform matrix
    warped_overlay = cv2.warpPerspective(resized_overlay, M, (base_image.shape[1], base_image.shape[0]))

    # Create a mask for the polygon region
    mask = np.zeros_like(base_image, dtype=np.uint8)
    cv2.fillPoly(mask, [rect_point], color=(255, 255, 255))

    # Use the mask to replace the region of interest in the base image with the warped overlay

    masked_base = cv2.bitwise_and(base_image, cv2.bitwise_not(mask))
    base_image = cv2.add(masked_base, warped_overlay)

  cv2.imwrite(image, cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
  return base_image


def resize_one_image_by_width(image_path, new_width, output_path):
   
   image = Image.open(image_path)

  #  main image size
   original_width, original_height = image.size 

  #  calculating the appropriate height
   new_height = int((new_width / original_width)* original_height)

  #  change size 
   resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
   resized_image.save(output_path)



# # test the model
# image = "/app/39.jpg"
# covered_image = cover_plate(model, image, logo)
# image_array = np.array(covered_image)
# plt.imshow(image_array)
# plt.axis('off')
# plt.show()


# Initialize Flask app and YOLO model
app = Flask(__name__)
model = YOLO("/app/best.pt")
logo = '/app/logo.png'

UPLOAD_FOLDER = 'tmp/uploads'
RESIZE_FOLDER = 'tmp/resized'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESIZE_FOLDER, exist_ok=True)

# Function definitions (transform_directions and cover_plate) remain unchanged

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/resize', methods=['POST'])
def resize_image():
   if 'image' not in request.files:
      return "No file sent.", 400
   
   file = request.files['image']
   if file.filename == '':
      return "The file name is empty." , 400
   
   new_width = int(request.form['width'])

  #  Save the uploaded file 
   filepath = os.path.join(UPLOAD_FOLDER, file.filename)
   file.save(filepath)

  #  set the output image path
   output_path = os.path.join(RESIZE_FOLDER, 'resized_image.jpg')

   try:

    #  resize the image and save it to the output path
    resize_one_image_by_width(filepath, new_width, output_path)


    return send_file(output_path, mimetype='image/jpeg', as_attachment=True, download_name='resized_image.jpg')
   except Exception as e:
      return jsonify({"error": str(e)}), 404

@app.route('/process', methods=['POST'])
def process_image():
    # Ensure an image file is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Save the uploaded file to a temporary location
    input_image_path = "/tmp/input_image.jpg"
    file.save(input_image_path)
    
    # Set the output image path
    output_image_path = "/tmp/output_image.jpg"
    
    try:
        # Process the image with YOLO model and overlay logo
        cover_plate(model, input_image_path, logo)
        
        # Move the processed image to the output path
        os.rename(input_image_path, output_image_path)
        
        # Return success message and link to download the image
        return send_file(output_image_path, mimetype='image/jpeg', as_attachment=True, download_name="processed_image.jpg")
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
