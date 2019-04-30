###
### script to detect objects in live video
### using MobileNets+SDD

import argparse, json
import time, datetime, threading
import numpy, cv2
import os, signal
from collections import deque

def read_frames(stream, queue):
    global detect
    while detect is True:
        (err, frame) = stream.read()
        queue.appendleft(frame)
    print('[INFO] exiting video stream thread...')

def execute_action(frame, timestamp, prediction, confidence, folder, action):
    # save frame to the 'detected' folder
    unixtime = (int)(time.mktime(timestamp.timetuple()))
    filename = '{0}/{1}_{2}.jpg'.format(folder, unixtime, prediction)
    cv2.imwrite(filename, frame)
    # execute action
    cmd = str(action).format(prediction, confidence, filename)
    os.system(cmd)

def exit_handler(signum, frame):
    global detect
    detect = False

### main code ###

# construct the argument parser and parse the program arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default='detect.json',
    help='path to configuration file')
args = vars(ap.parse_args())

# load configuration file
CONFIG_OPTIONS = [
    'prototxt',
    'caffemodel',
    'classes',
    'video_input',
    'video_output',
    'image_output',
    'batch_size',
    'base_confidence',
    'detect_classes',
    'detect_timeout',
    'detect_action',
    'statistics'
]
print('[INFO] loading configuration file...')
try:
    with open(args['config']) as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print('[ERROR] configuration file [{0}] not found!'.format(args['config']))
    exit(-1)

# detect missing configuration options
for option in CONFIG_OPTIONS:
    if option not in config:
        print('[ERROR] configuration option [{0}] not found in file [{1}]'.format(
            option, args['config']
        ))
        exit(-1)

# detect unknown configuration options
for option in config:
    if option not in CONFIG_OPTIONS:
        print('[WARNING] unknown configuration option [{0}]'.format(option))

# check image folder exists and create it if necessary
if not os.path.exists(config['image_output']):
    os.makedirs(config['image_output'])

# initialize the list of class labels MobileNets SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = config['classes']
COLORS = numpy.random.uniform(0, 255, size=(len(CLASSES), 3))

# load serialized model from disk
print('[INFO] loading caffemodel...')
try:
    open(config['prototxt'])
except FileNotFoundError:
    print('[ERROR] prototxt file [{0}] not found!'.format(config['prototxt']))
    exit(-1)
try:
    open(config['caffemodel'])
except FileNotFoundError:
    print('[ERROR] caffemodel file [{0}] not found!'.format(config['caffemodel']))
    exit(-1)
net = cv2.dnn.readNetFromCaffe(config['prototxt'], config['caffemodel'])

# initialize the input stream and allow the camera sensor to warmup
print('[INFO] connecting to video stream...')
vin = cv2.VideoCapture(config['video_input'])
time.sleep(2.0)

# detect video attributes and initialize the output stream
w = (int)(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
h = (int)(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vin.get(cv2.CAP_PROP_FPS)
print('[INFO] setting up '+str(w)+'x'+str(h)+'@'+str(fps)+' output stream...')
vout = cv2.VideoWriter(config['video_output'],0x21,fps, (w, h))

# initialize frames queue
batch_size = (int)(config['batch_size'])
queue = deque(maxlen=batch_size*2)

# start reading frames from video stream in separate thread
detect = True
reader = threading.Thread(name='reader', target=read_frames, args=(vin, queue,))
reader.start()

# install CTRL-C signal handler to handle graceful program exit
print('[INFO] installing CTRL-C handler...')
signal.signal(signal.SIGINT, exit_handler)

# loop over the frames from the video stream
print('[INFO] starting object detection...')
DETECTIONS = {}
STATISTICS = config['statistics']
processed = 0
start = datetime.datetime.now()
while detect is True:
    # grab a batch of frames from the threaded video stream
    frames = []
    for f in range(batch_size):
        while not queue:
            # wait for frames to arrive
            time.sleep(0.001)
        frames.append(queue.pop())

    if frames[0] is None:
        print('[ERROR] invalid frame received from input stream')
        detect = False
        continue

    # convert detection frame to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frames[0], (320, 320)), 0.007843, (320, 320), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    #dnn_start = time.time()
    detections = net.forward()
    #dnn_end = time.time()
    #print('[DEBUG] dnn detection took %0.3f ms' % ((dnn_end-dnn_start)*1000.0))

    # loop over the detections
    for i in numpy.arange(0, detections.shape[2]):
        # extract the prediction and the confidence (i.e., probability)
        # associated with the prediction
        obj = int(detections[0, 0, i, 1])
        prediction = CLASSES[obj]
        confidence = detections[0, 0, i, 2] * 100
        label = '{}: {:.2f}%'.format(prediction, confidence)

        # check if it's an object we are interested in
        # and if confidence is within the desired levels
        timestamp = datetime.datetime.now()
        detection_event = False
        if prediction in config['detect_classes']:
            if not confidence > (float)(config['detect_classes'][prediction]):
                # confidence too low for desired object
                continue
            else:
                # we detected something we are interested in
                # so we execute action assosciated with event
                # but only if the object class was not already detected recently
                if prediction in DETECTIONS:
                    prev_timestamp = DETECTIONS[prediction]
                    duration = (timestamp - prev_timestamp).total_seconds()
                    if duration > (float)(config['detect_timeout']):
                        # detection event (elapsed timestamp)
                        detection_event = True
                else:
                    # detection event (first occurence)
                    detection_event = True
        else:
            if not confidence > (float)(config['base_confidence']):
                # confidence too low for object
                continue

        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        tStartY = startY - 15 if startY - 15 > 15 else startY + 15

        # draw the prediction on the frame
        for frame in frames:
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[obj], 2)
            cv2.putText(frame, label, (startX, tStartY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[obj], 2)
#Added code for imshow
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
# if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # execute detection action
        if detection_event is True:
            DETECTIONS[prediction] = timestamp
            execute_action(frames[0], timestamp, prediction, confidence, config['image_output'], config['detect_action'])

    # write frames to output stream
    for frame in frames:
        vout.write(frame)
    if STATISTICS is True:
        processed += len(frames)

# do cleanup
end = datetime.datetime.now()
reader.join()
vin.release()
vout.release()

# display statistics
if STATISTICS is True:
    elapsed = (end - start).total_seconds()
    rfps = processed / elapsed
    print('[INFO] elapsed time: {:.2f} seconds'.format(elapsed))
    print('[INFO] approx. output FPS: {:.2f} (desired: {:.2f})'.format(rfps, fps))
    print('[INFO] approx. detection FPS: {:.2f}'.format(rfps/batch_size))
else:
    print('[INFO] exiting program...')
