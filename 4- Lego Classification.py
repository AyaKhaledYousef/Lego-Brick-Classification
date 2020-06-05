import cv2
import tensorflow as tf
import numpy as np
import glob as gb
import matplotlib.pyplot as plt  



def lenX_pred():
    predpath='lego_pred/'
    s=100
    global X_pred;
    X_pred = []
    files = gb.glob(pathname= str(predpath + 'lego_pred/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_pred.append(list(image_array))
    print(f'we have {len(X_pred)} items in X_pred')
# =============================================================================
# 
# =============================================================================
    
def XPredArray():
    global X_pred_array;
    X_pred_array = np.array(X_pred)
    print(f'X_pred shape  is {X_pred_array.shape}')
    X_pred_array=X_pred_array.astype('float32')
# =============================================================================
#     
# =============================================================================
def y_result(): 
    model = tf.keras.models.load_model("model2.h5")
    global y_result;
    y_result = model.predict(X_pred_array)
    print('Prediction Shape is {}'.format(y_result.shape))
# =============================================================================
# 
# =============================================================================
code = {'2x1':0 ,'2x2':1,'2x4':2}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return y

# =============================================================================
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    print("Capture photo ..")
    if cv2.waitKey(1):
        cv2.imwrite('lego_pred/lego_pred/capture.jpg', frame)
        lenX_pred()
        XPredArray()
        y_result()
        print(getcode(np.argmax(y_result[0])))

        break
cap.release()
cv2.destroyAllWindows() 