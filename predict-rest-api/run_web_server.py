# import numpy as np
import settings
from flask import Flask, request, jsonify
import redis
import uuid
import json
from copy import copy

# initialize our flask application and redis server
app = Flask(__name__)

# initialize all settings
s = settings.Settings()

db = redis.StrictRedis(host=s.REDIS_HOST, port=s.REDIS_PORT, db=s.REDIS_DB)


@app.route("/model")
def homepage():
	return "Main REST API!"


@app.route("/model/selectonly", methods=['POST'])
def selectonly():
	data = {"success": 'none'}
	uid = request.form['uid']
	target = request.form['target']
	iteration = request.form['iteration']
	dataset = request.form['dataset']

	d = {"id": uid, "uid": uid, "target": target, "iteration": iteration, "dataset": dataset}
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
	# db.ltrim(s.REQUEST_QUEUE, 1, -1)
	db.flushdb()
	return jsonify(data)


@app.route("/model/save", methods=['POST'])
def save():
	data = {"success": 'none'}
	uid = request.form['uid']
	target = request.form['target']
	classifier = request.form['classifier']
	posclass = request.form['posclass']
	negclass = request.form['negclass']
	iteration = request.form['iteration']
	dataset = request.form['dataset']
	reloaded = request.form['reloaded']

	d = {"id": uid, "uid": uid, "target": target,
	"classifier": classifier, "reloaded": reloaded,
	"posclass": posclass, "negclass": negclass,
	"iteration": iteration, "dataset": dataset}

	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
	# db.ltrim(s.REQUEST_QUEUE, 1, -1)
	db.flushdb()
	return jsonify(data)


@app.route("/model/view", methods=['POST'])
def view():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)


@app.route("/model/retrainView", methods=['POST'])
def retrainView():
	data_retrainview = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data_retrainview = copy(output)
			db.delete(uid)
			break
		# data["success"] = True
	db.flushdb()
	return jsonify(data_retrainview)


@app.route("/model/retrainHeatmap", methods=['POST'])
def retrainHeatmap():
	data_retrainheatmap = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data_retrainheatmap = copy(output)
			db.delete(uid)
			break
			# time.sleep(s.SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data_retrainheatmap)



@app.route("/model/reload", methods=['POST'])
def reload():
	data = {"success": 'fail'}
	uid = request.form['uid']
	target = request.form['target']
	dataset = request.form['dataset']
	trainingSetName = request.form['trainingSetName']

	d = {"id": uid, "uid": uid, "target": target,
	"trainingSetName": trainingSetName, "dataset": dataset
	}

	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			data = copy(output)
			break

	db.flushdb()
	# time.sleep(s.SLEEP)
	return jsonify(data)


@app.route("/model/train", methods=['POST'])
def train():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			db.delete(uid)
			break
		# data["success"] = True
	db.flushdb()
	return jsonify(data)

@app.route("/model/heatmap", methods=['POST'])
def heatmap():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)


@app.route("/model/heatmapAll", methods=['POST'])
def heatmapAll():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)


@app.route("/model/review", methods=['POST'])
def review():
	data = {"success": 'none'}
	uid = request.form['uid']
	target = request.form['target']
	dataset = request.form['dataset']

	d = {"id": uid, "uid": uid, "target": target,
		"dataset": dataset}

	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)


@app.route("/model/reviewSave", methods=['POST'])
def reviewSave():
	data = {"success": 'none'}
	uid = request.form['uid']
	target = request.form['target']
	samples = request.form['samples']
	dataset = request.form['dataset']

	d = {"id": uid, "uid": uid, "target": target,
	"samples": samples, "dataset": dataset
	}

	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)



@app.route("/model/cancel", methods=['POST'])
def cancel():
	data = {"success": 'none'}
	uid = request.form['uid']
	target = request.form['target']
	dataset = request.form['dataset']

	d = {"id": uid, "uid": uid, "target": target,
		"dataset": dataset}

	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			# output = output.decode("utf-8")
			# data = json.loads(output)
			data = copy(output)
			# db.delete(uid)
			break
			# time.sleep(s.CLIENT_SLEEP)
		# data["success"] = True
	db.flushdb()
	return jsonify(data)


@app.route("/model/label", methods=['POST'])
def label():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			data = copy(output)
			break
	db.flushdb()
	return jsonify(data)

@app.route("/model/count", methods=['POST'])
def count():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			data = copy(output)
			break
	db.flushdb()
	return jsonify(data)

@app.route("/model/map", methods=['POST'])
def map():
	data = {"success": 'none'}
	d = json.loads(request.data)
	uid = d.get('uid')
	db.rpush(s.REQUEST_QUEUE, json.dumps(d))
	while True:
		output = db.get(uid)
		if output is not None:
			data = copy(output)
			break
	db.flushdb()
	return jsonify(data)



if __name__ == "__main__":
	app.run()
