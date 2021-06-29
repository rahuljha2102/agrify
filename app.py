from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


app=Flask(__name__,template_folder='/content')

run_with_ngrok(app)


@app.route('/', methods =["GET", "POST"])
def home():
  if request.method == "POST":
    global crop_selection
    global a1
    crop_selection= request.form.get("crops")
    
    if(crop_selection=='Potato'):
      a1='potato_index.html'

    elif(crop_selection=='Grapes'):
      a1='grape_index.html'

    elif(crop_selection=='Apple'):
      a1='apple_index.html'
 
    elif(crop_selection=='Tomato'):
      a1='tomato_index.html'
    return render_template(a1)
  return render_template('main.html')
  
#---------------------------------------------------------------------

#Import necessary libraries


#----------------------------------------------------------------------------











model3=load_model("/content/potato_data.h5")
 
print('@@ Model loaded')


 
 
def pred_potato_disease(potato_plant):
  test_image = load_img(potato_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model3.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return 'POTATO EARLY BLIGHT','potato_early_blight.html'
  elif pred == 1:
    return 'POTATO LATE BLIGHT','potato_late_blight.html'
  elif pred == 2:
    return 'POTATO HEALTHY','potato_healthy.html'


@app.route("/potato_predict", methods = ['GET','POST'])
def predict_potato():
  if request.method == 'POST':
    file = request.files['potato_image'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
         
    file_path = os.path.join('/static/user_uploaded', filename)
    file.save(file_path)
 
    print("@@ Predicting class......")
    pred, output_page = pred_potato_disease(potato_plant=file_path)
               
    return render_template(output_page, pred_output = pred, user_image = file_path)
     








model1=load_model("/content/grape.h5")
 
print('@@ Model loaded')
 
 
def pred_grape_disease(grape_plant):
  test_image = load_img(grape_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model1.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return 'Grape Black Rot','grape_black_rot.html'
  elif pred == 1:
    return 'Grape Esca (Black_Measles)','grapes_black_meales.html'
  elif pred == 2:
    return 'Grape Leaf Blight_(Isariopsis_Leaf_Spot)','grape_leaf_blight.html'
  elif pred == 3:
    return 'Grape Healthy','grape_healthy.html'
#------------>>pred_cot_dieas<<--end
     
 
# Create flask instance

 
# render index.html page

     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict_grape", methods = ['GET','POST'])
def predict_grape():
  if request.method == 'POST':
    file = request.files['grape_image'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
         
    file_path = os.path.join('/static/user_uploaded', filename)
    file.save(file_path)
 
    print("@@ Predicting class......")
    pred, output_page = pred_grape_disease(grape_plant=file_path)
               
    return render_template(output_page, pred_output = pred, user_image = file_path)







     

model2=load_model("/content/tomato (1).h5")
 
print('@@ Model loaded')
 
 
def pred_tomato_disease(tomato_plant):
  test_image = load_img(tomato_plant, target_size = (256,256)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model2.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return 'Tomato Bacterial spot','tomato_bacterial_spot.html'
  elif pred == 1:
    return 'Tomato Early Blight','tomato_early_blight.html'
  elif pred == 2:
    return 'tomato Late Blight','tomato_late_blight.html'
  elif pred == 3:
    return 'Tomato Leaf Mold','tomato_leaf_mold.html'
  elif pred == 4:
    return 'Tomato Septoria leaf spot','tomato_Treat _Septoria_Leaf_Spot.html'
  elif pred == 5:
    return 'Tomato Spider mites Two-spotted_spider_mite','tomato_two-spotted Spider Mite.html'
  elif pred ==6:
    return 'Tomato Target Spot','tomato_target_spot.html'
  elif pred == 7:
    return 'Tomato Tomato Yellow Leaf Curl Virus','Tomato_yellow_leaf_curl_virus.html'
  elif pred == 8:
    return 'Tomato Tomato mosaic virus','tomato_mosaic.html'
  elif pred ==9:
    return 'Tomato Healthy Plant','tomato_healthy.html'
 
#------------>>pred_cot_dieas<<--end
     
 
# Create flask instance

 
# render index.html page

     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict_tomato", methods = ['GET','POST'])
def predict_tomato():
  if request.method == 'POST':
    file = request.files['tomato_image'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
         
    file_path = os.path.join('/static/user_uploaded', filename)
    file.save(file_path)
 
    print("@@ Predicting class......")
    pred, output_page = pred_tomato_disease(tomato_plant=file_path)
               
    return render_template(output_page, pred_output = pred, user_image = file_path)
     



     







app.run()
