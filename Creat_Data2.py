import cv2
import os
import time
def data1 (dir1):
    img_counter=1
    while(True):
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        fps=cap.get(4)
        
        
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
        
        
        cv2.imshow('frame', rgb)
        cap.release()
        
        #directory=os.path.abspath(os.path.join(os.path.curdir))
    
        filename=dir1 +"image_{}.jpg".format(img_counter)
        img_counter +=1
    
    
        cv2.imwrite(filename, frame)
        
        k=cv2.waitKey(1)
        
        if k%256==27:
            #Esc pressed
            print("Escape .... ")
            break
    
    cap.release()
    cv2.destroyAllWindows()
sleep=2  
directory=os.path.abspath(os.path.join(os.path.curdir))   
data1(directory+"/lego_train/lego_train/2x2/")
time.sleep(sleep)
data1(directory+"/lego_train/lego_train/2x3/")
time.sleep(sleep)
data1(directory+"/lego_train/lego_train/2x4/")
time.sleep(sleep)
data1(directory+"/lego_test/lego_test/2x2/")
time.sleep(sleep)
data1(directory+"/lego_test/lego_test/2x3/")
time.sleep(sleep)
data1(directory+"/lego_test/lego_test/2x4/")

# =============================================================================
# Data augmentation
# =============================================================================
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
def data_gen(dir1,dir2):
    img = load_img(dir1)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(rescale=1./255,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        validation_split=0.1)
    # prepare iterator
    img_counter=1
    it = datagen.flow(samples, batch_size=1,save_to_dir=dir2, save_prefix="image_gen_{}".format(img_counter),save_format='jpg')
    img_counter +=1
    # generate samples and plot
    for i in range(9):
    	# define subplot
    	pyplot.subplot(330 + 1 + i)
    	# generate batch of images
    	batch = it.next()
    	# convert to unsigned integers for viewing
    	image = batch[0].astype('uint8')
    	# plot raw pixel data
    	pyplot.imshow(image)
    # show the figure
    pyplot.show()

# =============================================================================
D1='lego_train/lego_train/2x2'
data_gen('lego_train/lego_train/2x2/image_1.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_2.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_3.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_4.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_5.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_6.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_7.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_7.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_8.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_9.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_10.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_11.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_12.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_13.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_14.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_15.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_16.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_17.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_18.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_19.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_20.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_21.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_22.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_23.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_24.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_25.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_26.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_27.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_28.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_29.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_30.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_31.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_32.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_33.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_34.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_35.jpg',D1)
data_gen('lego_train/lego_train/2x2/image_36.jpg',D1)


# =============================================================================
# 2x3
# =============================================================================
D2='lego_train/lego_train/2x3'
data_gen('lego_train/lego_train/2x3/image_1.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_2.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_3.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_4.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_5.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_6.jpg',D2)
data_gen('lego_train/lego_train/2x3/image_7.jpg',D2)
# =============================================================================
# 2x4
# =============================================================================
D3='lego_train/lego_train/2x4'
data_gen('lego_train/lego_train/2x4/image_1.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_2.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_3.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_4.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_5.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_6.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_7.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_8.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_9.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_10.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_11.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_12.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_13.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_14.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_15.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_16.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_17.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_18.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_19.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_20.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_21.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_22.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_23.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_24.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_25.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_26.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_27.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_28.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_29.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_30.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_31.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_32.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_33.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_34.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_35.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_36.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_37.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_38.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_39.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_40.jpg',D3)
data_gen('lego_train/lego_train/2x4/image_41.jpg',D3)


