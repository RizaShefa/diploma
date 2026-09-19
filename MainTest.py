import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BreastCancer10Epochs.h5')

image=cv2.imread(r'C:\Users\Enea\Desktop\ultrasound breast classification\pred\pred (600).png')



img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)

input_img=np.expand_dims(img,axis=0)

result=np.argmax(model.predict(input_img),axis=1)
print(result)
