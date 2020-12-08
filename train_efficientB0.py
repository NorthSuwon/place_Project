import sys, os, numpy, h5py
from modules import preprocessing, build_model
from modules.load_image import DataGenerator
from keras import optimizers, losses, metrics
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split

def warn(*args, **kwargs):
	pass

def main():
    pre = preprocessing()
    images, labels = pre.__getimage__()
	X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.1, random_state=321)
	
	model=build_model(1049)
	check_point = ModelCheckpoint("Weight/eff013_{epoch:03d}.h5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=100, verbose=1, mode='auto', restore_best_weights=True)
	history0 = model.fit(X_train, Y_train,
						validation_data=(X_test, Y_test), 
						callbacks=[check_point, monitor], batch_size=32, epochs=20, verbose=1)
	model.save_weights('last_weight.h5')

	
if __name__ == '__main__':
	try:
		main()
		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interupted or Error!!!")