from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


app=Flask(__name__)

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











model3=load_model("potato_data.h5")
 
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
         
    file_path = os.path.join('/user_uploaded', filename)
    file.save(file_path)
 
    print("@@ Predicting class......")
    pred, output_page = pred_potato_disease(potato_plant=file_path)
               
    return render_template(output_page, pred_output = pred, user_image = file_path)
     



if __name__ == "__main__":
    app.run(debug=True)





