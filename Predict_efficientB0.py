import h5py
import os,csv,cv2
import numpy
import sys
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from modules.load_image import image_load
from keras import optimizers, losses, metrics
from keras.callbacks import EarlyStopping

def main():	
    path = "U:\\North_Suwon\\Data\\"
    path_test = path+"test\\"
    a=image_load(path)
    model = build_model(1049)
    weight_location ='U:\\North_Suwon\\Weight\\b0/eff_0013_100.h5'
    model.load_weights(weight_location)
    f = open('sample_submisstion.csv', 'r', encoding='utf-8') # 파일열기
    fw = open('outputssaas.csv', 'w', encoding='utf-8',newline="") # 읽기

    wr = csv.writer(fw)
    wr.writerow(["id", "landmark_id", "conf"])
    rdr = csv.reader(f)
    next(rdr)
    f_name=[];answer =[]; acc = []

    for line in rdr:
        f_name.append(line[0])
        print(line[0])

    h5f = h5py.File("test_set.h5", 'r')
    images = h5f['images'][:]
    h5f.close()

    
    prediction = model.predict(images,verbose=1)
    
    for i in range(len(f_name)):
        arridx = numpy.where(prediction[i]==max(prediction[i]))
        print(arridx[0][0])
        wr.writerow([f_name[i], arridx[0][0], max(prediction[i])])
    
    f.close()   
    fw.close()
    
if __name__ == '__main__':
	try:
		main()
		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interrupted or Error!!!")
