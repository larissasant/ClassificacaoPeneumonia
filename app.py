from PIL.Image import Image
from flask import Flask, render_template, request
from keras.backend import batch_dot
from keras.models import load_model
from keras.preprocessing import image

import numpy as np

app = Flask(__name__)

#dic = {0 : 'normal', 1 : 'pneumonia'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	img = image.load_img(img_path, target_size=(300,300))
	i = image.img_to_array(img)
	i = np.expand_dims(i, axis=0)

	images = np.vstack([i])
	p = model.predict(images, batch_size = 10)

	print("classe", p[0])
	
	if p[0] > 0.5:
		print('Pneumonia')
		return p[0]
	else:
		print('Normal')
		return p[0]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Trabalho desenvolvido para conclusão de curso"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/"+img.filename	
		img.save(img_path)

		p = predict_label(img_path)

		print("teste", img_path)
		print("teste2", p)

	return render_template("index.html", pred = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)