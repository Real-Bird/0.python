from tensorflow import keras
model = keras.models.load_model("tn_model.h5", compile=False)

export_path = './tn_model/'
model.save(export_path, save_format='tf')