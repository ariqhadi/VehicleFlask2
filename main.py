import cv2
import numpy as np
import os
import time
from datetime import datetime,timedelta
import tensorflow as tf
import base64

from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import threading
import socketio
import eventlet
from socketio import server
from eventlet import wsgi, patcher
import json

from yolo_v3 import Yolo_v3
from utils_tf2 import load_class_names, draw_frame, get_roi_frame
from sort import *
from line import *
from inputText import printHasil
from capacityCounting import PipelineRunner,CapacityCounter
from saveFile import saveFile, exportJson

lock = threading.Lock()
sio = server
app = Flask(__name__)
CORS(app)

line = DirectionTracker()

#Initialize background Image for Capacity Counter
base_bgrd= cv2.imread("/home/kp_polban/research/Vehicle Counting_Traffic_Optimized/base background/Relaxing highway traffic.jpg")

#Initialize Video dan penghitung durasi video
video = '/home/kp_polban/research/Vehicle Counting_Traffic_Optimized/input/Relaxing highway traffic.mp4'

#inisiasi Video
cap = cv2.VideoCapture(video)
frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
jmlFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
durasi = int(jmlFrame/fps)
timePerFrame = durasi/jmlFrame
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
f_height, f_width, _ = frame.shape #tinggi dan lebar video
shape = (f_height,f_width)

# Initialize Output File
outs = cv2.VideoWriter('/home/kp_polban/research/Vehicle Counting_Traffic_Optimized/output/Relaxing highway traffic - 30 FPS.avi',cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (f_width,f_height))

#Initialize Log File
logFile = "/home/kp_polban/research/Vehicle Counting_Traffic_Optimized/output/Relaxing highway traffic - 30 FPS.json"

# Inisisasi Objek Pipeline (untuk deteksi kapasitas)
pipeline = PipelineRunner(pipeline=[ CapacityCounter()])

#inisialisasi variabel
durationInVideo = 0

# Inisialisasi Waktu
waktu11 = time.time()


frame_counter = 0 
DETECTION_INTERVAL = 2 #berapa frame sekali video di proses
timeout = time.time() + 60


# create detection Region of Interest _ROI
droi2 =[(int(f_width/2)+150, 350), (int(f_width/2)+20, 350), (int(f_width/2)+10, f_height), (f_width, f_height-100)]  #tracking hanya bekerja dalam ROI
droi1 =  [(0, int(f_height/2)), (f_width, int(f_height/2)), (f_width, f_height), (0, f_height)]
droi_all = [(0,f_height),(f_width,f_height),(f_width,0),(0,0)]


#Initialize Tracker
mot_tracker = Sort()


print("jaksfjahkdfj")
_MODEL_SIZE = (416, 416)
class_names = load_class_names("weights/vehicle.names")
n_classes = len(class_names)
_MAX_OUTPUT_SIZE = 20

model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                max_output_size=_MAX_OUTPUT_SIZE,
                iou_threshold=0.5,
                confidence_threshold=0.5)

inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
detections = model(inputs, training=False)
saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))


def generate():

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')
        
        while True:        
            k = cv2.waitKey(1)
            ret,frame = cap.read()
            if ret:
                t0= time.time()

                pipeline.set_context({
                'frame': frame,
                'frame_number':frame_counter,
                'base_bgrd':base_bgrd
                })

                context = pipeline.run()

                i = 0

                # rerun detection
                # if frame_counter > DETECTION_INTERVAL:
                # frame_counter = 0

                droi_frame = get_roi_frame(frame, droi1)
                resized_frame = cv2.resize(droi_frame, dsize=_MODEL_SIZE[::-1],
                                                interpolation=cv2.INTER_NEAREST)
                detection_result = sess.run(detections,
                                        feed_dict={inputs: [resized_frame]})
                detected_classes, klass = draw_frame(droi_frame, frame_size, detection_result,
                                                class_names, _MODEL_SIZE)
                

                tracks = mot_tracker.update(np.asarray(klass))
                line.vehicle_track(frame,tracks,droi1,detected_classes)
                
                # durationInVideo = durationInVideo + timePerFrame
                printHasil(line.maju,line.mundur,frame,str('%.1f'%(context['capacity1'])),str('%.1f'%(context['capacity2'])))
                # infoJson = exportJson(logFile,todayDate,durasi,line.maju,line.mundur,kapasitas1,kapasitas2,jam )
                # frame_counter += 1
                
                todayDate = datetime.now()
                
                # cv2.imshow("hasila",resized_frame)
                # outs.write(frame)
                # visual = np.vstack((frames[0:]))
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                
                # if(time.time()>timeout):
                kapasitas1 = str('%.1f'%(context['capacity1']))
                kapasitas2 = str('%.1f'%(context['capacity2']))
                durasi = str(timedelta(seconds=int(durationInVideo)))
                jam = time.time()-waktu11
                infoJson = exportJson(logFile,todayDate,durasi,line.maju,line.mundur,kapasitas1,kapasitas2,jam )
                #     saveFile(logFile,todayDate,durasi,line.maju,line.mundur,kapasitas1,kapasitas2,jam)
                    
                #     timeout = time.time()+60
                
        #         print("WAKTU PROSES PER FRAME : ",(time.time()-t0))
                print(infoJson)
                # infoJson.update({'image':base64.b64encode(frame)})
            else:
                print('End of video.')
                break
            # print(infoJson)
            yield json.dumps(infoJson)
            # yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n') 

        #     # end video loop if 'q' key is pressed
        #     if k & 0xFF == ord('q'):
        #         print('Video exited.')
        #         break

        # cap.release()
        # outs.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
		mimetype = "application/json")


@app.route("/")
def index():
	return render_template('index.html')

if __name__ == '__main__':
	app = socketio.Middleware(sio, app)
	wsgi.server(eventlet.listen(('', 9100)), app)
	patcher.monkey_patch(all=False, socket=True, time=True,
		select=True, thread=True, os=True)
