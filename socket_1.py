from flask import Flask, request, jsonify
from datetime import datetime
import pymongo
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from IPython.display import display
from flask_socketio import SocketIO
from flask_socketio import send, emit

app = Flask(__name__)
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
client = MongoClient("mongodb+srv://musa:1234@cluster0.ps9aijg.mongodb.net/test")
db = client.users
grid_fs = gridfs.GridFS(db)



@socketio.on('message')
def handle_json(message):
    print('received json: ' + str(message))

@socketio.on('message')
def handle_json(message):
    send(message)

@socketio.on('send_message')
def handle_source(json_data):
    emit('data', {'data': json_data }, broadcast=True, include_self=False)

if __name__ == "__main__":
    socketio.run(app, port=10001, debug=True)


# pip install eventlet
# gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 --bind 0.0.0.0:10001 socket_1:app