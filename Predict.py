import numpy as np
from keras.models import model_from_json
import operator
import cv2

#loading the model
json_file=open('face_emotions-model.json','r')
model_json=json_file.read()
json_file.close()

loaded_model=model_from_json(model_json)

#load weights into model
loaded_model.load_weights('face_emotions_weights.h5')
print('Loaded model from disk')

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame = cv2.flip(frame, 1)

    #coordinates of ROI
    cv2.rectangle(frame, (175, 100), (500, 360), (0, 255, 0), 2)
    #extracing the ROI
    roi=frame[100:360, 175:500]
    roi=cv2.resize(roi,(64,64))
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)
    # kernel=np.ones((2,2),np.uint8)
    # roi=cv2.dilate(roi,kernel,iterations=2)
    cv2.imshow('roi', roi)

    #batch of 1
    result=loaded_model.predict(roi.reshape(1,64,64,1))

    prediction={'Calm':result[0][0],
                 'Happy':result[0][1],
                 'Yawning':result[0][2]}

    #sorting the predctions based on top predictions
    prediction=sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    #display the prediction
    cv2.putText(frame,prediction[0][0], (10,120), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    cv2.imshow('frame',frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()
