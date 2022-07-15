import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import ImageFile
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from extract_bottleneck_features import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

dog_names = ['Affenpinsche', 'Afghan_houn', 'Airedale_terrie', 'Akit', 'Alaskan_malamut', 'American_eskimo_do', 'American_foxhoun', 'American_staffordshire_terrie', 'American_water_spanie', 'Anatolian_shepherd_do', 'Australian_cattle_do', 'Australian_shepher', 'Australian_terrie', 'Basenj', 'Basset_houn', 'Beagl', 'Bearded_colli', 'Beaucero', 'Bedlington_terrie', 'Belgian_malinoi', 'Belgian_sheepdo', 'Belgian_tervure', 'Bernese_mountain_do', 'Bichon_fris', 'Black_and_tan_coonhoun', 'Black_russian_terrie', 'Bloodhoun', 'Bluetick_coonhoun', 'Border_colli', 'Border_terrie', 'Borzo', 'Boston_terrie', 'Bouvier_des_flandre', 'Boxe', 'Boykin_spanie', 'Briar', 'Brittan', 'Brussels_griffo', 'Bull_terrie', 'Bulldo', 'Bullmastif', 'Cairn_terrie', 'Canaan_do', 'Cane_cors', 'Cardigan_welsh_corg', 'Cavalier_king_charles_spanie', 'Chesapeake_bay_retrieve', 'Chihuahu', 'Chinese_creste', 'Chinese_shar-pe', 'Chow_cho', 'Clumber_spanie', 'Cocker_spanie', 'Colli', 'Curly-coated_retrieve', 'Dachshun', 'Dalmatia', 'Dandie_dinmont_terrie', 'Doberman_pinsche', 'Dogue_de_bordeau', 'English_cocker_spanie', 'English_sette', 'English_springer_spanie', 'English_toy_spanie', 'Entlebucher_mountain_do', 'Field_spanie', 'Finnish_spit',
             'Flat-coated_retrieve', 'French_bulldo', 'German_pinsche', 'German_shepherd_do', 'German_shorthaired_pointe', 'German_wirehaired_pointe', 'Giant_schnauze', 'Glen_of_imaal_terrie', 'Golden_retrieve', 'Gordon_sette', 'Great_dan', 'Great_pyrenee', 'Greater_swiss_mountain_do', 'Greyhoun', 'Havanes', 'Ibizan_houn', 'Icelandic_sheepdo', 'Irish_red_and_white_sette', 'Irish_sette', 'Irish_terrie', 'Irish_water_spanie', 'Irish_wolfhoun', 'Italian_greyhoun', 'Japanese_chi', 'Keeshon', 'Kerry_blue_terrie', 'Komondo', 'Kuvas', 'Labrador_retrieve', 'Lakeland_terrie', 'Leonberge', 'Lhasa_aps', 'Lowche', 'Maltes', 'Manchester_terrie', 'Mastif', 'Miniature_schnauze', 'Neapolitan_mastif', 'Newfoundlan', 'Norfolk_terrie', 'Norwegian_buhun', 'Norwegian_elkhoun', 'Norwegian_lundehun', 'Norwich_terrie', 'Nova_scotia_duck_tolling_retrieve', 'Old_english_sheepdo', 'Otterhoun', 'Papillo', 'Parson_russell_terrie', 'Pekinges', 'Pembroke_welsh_corg', 'Petit_basset_griffon_vendee', 'Pharaoh_houn', 'Plot', 'Pointe', 'Pomerania', 'Poodl', 'Portuguese_water_do', 'Saint_bernar', 'Silky_terrie', 'Smooth_fox_terrie', 'Tibetan_mastif', 'Welsh_springer_spanie', 'Wirehaired_pointing_griffo', 'Xoloitzcuintl', 'Yorkshire_terrie']


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

ResNet50_model = ResNet50(weights='imagenet')
resnet_model = Sequential()
resnet_model.add(GlobalAveragePooling2D(input_shape=[7, 7, 2048]))
resnet_model.add(Dense(133, activation='softmax'))
resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
resnet_model.load_weights('saved_models/weights.best.resnet.hdf5')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)



def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # prediction
    predict = resnet_model.predict(bottleneck_feature)
    # return predicted dog breed
    return dog_names[np.argmax(predict)]





def breed_algorithm(img_path):

    if face_detector(img_path):
        return 0, predict_breed(img_path)

    elif dog_detector(img_path):
        return 1, predict_breed(img_path)

    else:
        return 2, predict_breed(img_path)
