import os,csv
import cv2
import numpy
import h5py

class preprocessing():
    def __init__(self):
        pass

    @staticmethod
    def contrast_limited_adaptive_HE(channel_img):
        assert(len(channel_img.shape)==2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))      # Create CLAHE Object
        clahe_image = numpy.empty(channel_img.shape, dtype='uint8')
        clahe_image = clahe.apply(numpy.array(channel_img, dtype='uint8'))
        return clahe_image

    def image_preprocessing(self, img):
        #Display.Im3D(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)

        l = self.contrast_limited_adaptive_HE(l)

        processed_image = cv2.merge((l, a, b))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2RGB)
        return processed_image

    def __getIndex__(self):
        # Make Dictionary
        key_val = dict()
        f = open('train.csv', 'r', encoding='utf-8') # 파일열기
        rdr = csv.reader(f)
        temp_Name = ""
        for line in rdr:
            if temp_Name != line[0][:-4]:
                key_val[line[0][:-4]] = int(line[1])
                temp_Name=line[0]
        f.close()

        return key_val

    def __getimage__(self):        
        path = "/public\\"
        path_train = path+"train\\"
        key_val = self.__getIndex__(self)

        # Read images
        images = []; labels = []
        _types = next(os.walk(path_train))[1] # 광역시 읽기 path = ./public/train/
        img=[]
        for _type in _types: # Big City
            _places = next(os.walk(path_train+_type))[1] #
            for _place in _places:
                _place_imgs = next(os.walk(path_train+_type+"/"+_place))[2:] # Image name
                for _place_img in _place_imgs:
                    for _img in _place_img:
                        labels.append(key_val.get(_img[:-8]))
                        sub=path_train+_type+"/"+_place+"/"+_img
                        print(sub,key_val.get(_img[:-8]))
                        n = numpy.fromfile(sub, numpy.uint8)
                        _img = cv2.imdecode(n, cv2.IMREAD_COLOR)
                        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                        _img = cv2.resize(_img, (224, 224))
                        _img = self.image_preprocessing(_img)
                        img.append(_img)

        # Get all images in Folders
        img = numpy.array(img, dtype="uint8")
        labels = numpy.array(labels, dtype='uint16')
        
        return img, labels

    @staticmethod
    def __save__(images, labels, FileName='image_data_aaa.h5'):
        # Save images and theirs labels to file
        h5f = h5py.File(FileName, 'a')

        h5f.create_dataset('images', data=images)
        h5f.create_dataset('labels', data=labels)

        h5f.close()

    @staticmethod
    def on_epoch_end():
        pass

    def __len__(self):
        return int(numpy.ceil(len(self.IDs)))