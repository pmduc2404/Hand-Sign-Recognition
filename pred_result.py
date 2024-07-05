import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

label = ['0','1','A','B','C','D','Confirm','E','F','G','Clear','H','2','3','4','5','6','7','8','9']

model = load_model('./best_model.h5')



def get_prediction(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    # Get the predicted value
    pred_value = model.predict(x)
    
    # Check if the predicted value is greater than 0.8
    if np.any(pred_value > 0.8):
        pred = str(label[np.argmax(pred_value)])
        print(np.argmax(pred_value))
        return pred
    else:
        return "Khong khop"
