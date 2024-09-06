from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array
from tensorflow.keras.utils import load_img
dat_gen=ImageDataGenerator(
    rotation_range=20,  
    horizontal_flip=True,  
    fill_mode='nearest'
)

img = load_img("dataset/12440017704_068016572a_o.jpg")
x=img_to_array(img)
x=x.reshape((1,)+x.shape)

i=0
for batch in dat_gen.flow(x,batch_size=1,save_prefix='i19',save_to_dir='dataset',save_format='jpeg'):
    i+=1
    if i>20:
        break
