from flask import Flask, render_template, g, jsonify, request
import json
import numpy as np
import codecs
import zmq

app = Flask(__name__, static_url_path='/static')

MOVIE_NAME_PATH = "../ml-100k/u.item"

MOVIE_LIKE_PATH = '../ml-100k/demo.data'

MOVIE_GENRES=['unknown', 'ACTION', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', \
	'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:%s" % port)

def read_movie_names(filename):
	movies = []
	with codecs.open(filename, encoding='latin-1') as f:
		for line in f:
			line = line[:-1].split("|")
			movies.append({
				'id': int(line[0]),
				"title": line[1],
				"release": line[2],
				"video_release": line[3],
				"URL": line[4],
				"genre": MOVIE_GENRES[np.argmax(map(int,line[5:]))]
			})

	return movies

@app.route("/")
def hello():

	return render_template("index.html")

@app.before_request
def make_movies():
	if not hasattr(g, "movies"):
		g.movies = read_movie_names(MOVIE_NAME_PATH)

@app.route("/movies", methods=['POST'])
def movie_request():

	return jsonify(g.movies)

@app.route("/predict", methods=['POST'])
def movie_predict():

	entries = json.loads(request.form.get("entries"))

	print entries

	with open(MOVIE_LIKE_PATH, "w+") as f:
		for entry in entries:
			f.write("%d\t%d\t%d\t%d\n"%(entry["userid"], entry["movie_id"], entry["rating"], entry["time"]))

	socket.send_json({
		"type": "request",
		"path": MOVIE_LIKE_PATH
	})

	top_movies = socket.recv_json()["prediction"]

	return jsonify(top_movies)

if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True)