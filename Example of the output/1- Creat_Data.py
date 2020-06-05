# =============================================================================
# Import necessary liberaries 
# =============================================================================
import cv2
import os
import time
# =============================================================================
# Take automatic pic in each 2 sec until you press Esc
# =============================================================================
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
# save first group  in lego train 2x2
data1(directory+"/lego_train/lego_train/2x2/")
# wait 2 sec
time.sleep(sleep)
# save first group  in lego train 2x1
data1(directory+"/lego_train/lego_train/2x1/")
# wait 2 sec
time.sleep(sleep)
# save first group  in lego train 2x4
data1(directory+"/lego_train/lego_train/2x4/")
# wait 2 sec
time.sleep(sleep)
# save first group  in lego test 2x2
data1(directory+"/lego_test/lego_test/2x2/")
# wait 2 sec
time.sleep(sleep)
# save first group  in lego test 2x1
data1(directory+"/lego_test/lego_test/2x1/")
# wait 2 sec
time.sleep(sleep)
# save first group  in lego test 2x4
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
# 2x2
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
# =============================================================================
# 2x1
# =============================================================================
D2='lego_train/lego_train/2x1'
data_gen('lego_train/lego_train/2x1/image_1.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_2.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_3.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_4.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_5.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_6.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_7.jpg',D2)
data_gen('lego_train/lego_train/2x1/image_8.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_9.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_10.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_11.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_12.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_13.jpg',D3)
data_gen('lego_train/lego_train/2x1/image_14.jpg',D3)
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

