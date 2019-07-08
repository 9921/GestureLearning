#! /usr/bin/env python3

import copy
import cv2
import numpy as np
from keras.models import load_model
from phue import Bridge
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import rospy #importar ros para python
from std_msgs.msg import String, Int32 # importar mensajes de ROS tipo String y tipo Int32
from geometry_msgs.msg import Twist # importar mensajes de ROS tipo geometry / Twist
from duckietown_msgs.msg import Twist2DStamped #mensajes

# Ajustes generales
prediction = ''
action = ''
score = 0
img_counter = 500

on_command = {'transitiontime': 0, 'on': True, 'bri': 254}
off_command = {'transitiontime': 0, 'on': False, 'bri': 254}

# diccionario de gestos
gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

# se carga la red neuronal
model = load_model('VGG_cross_validated.h5')

# función que entrega la predicción
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print('pred_array: {}'.format(pred_array))
    result = gesture_names[np.argmax(pred_array[0])]
    print('Result: {}'.format(result))
    print(result)
    score = float("%0.2f" % (np.argmax(pred_array) * 100))
    print(score)
    return result, score

class Template(object):
    def __init__(self, args):
        super(Template, self).__init__()
        self.args = args
        self.publisher = rospy.Publisher('/duckiebot/wheels_driver_node/car_cmd', Twist2DStamped, queue_size = 1)
        self.twist = Twist2DStamped()
        self.isBgCaptured = 0  # bool, si se capturó el background
        self.triggerSwitch = False  # si es true, el simulador de teclado funciona
        self.bgModel = None


    # función que obtiene la predicción
    # dependiendo de ella, setea los valores de la velocidad y el ángulo
    def predicion(self, thresh):
        # copia una imagen en blanco y negro a los 3 canales RGB
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_rgb_image_vgg(target)


        if prediction == 'Palm':
            rospy.loginfo("Derecha")
            self.twist.v = 8
            self.twist.omega = 150

        elif prediction == 'Fist':
            rospy.loginfo("Parar")
            self.twist.v = 0
            self.twist.omega = 0

        elif prediction == 'L':
            rospy.loginfo("Izquierda")
            self.twist.omega = -150
            self.twist.v = 8

        elif prediction == 'Okay':
            rospy.loginfo("Adelante")
            self.twist.omega = 0
            self.twist.v= 8

        elif prediction == 'Peace':
            rospy.loginfo("Atrás")
            self.twist.omega = 0
            self.twist.v = -8

    # función que setea el background
    def set_background(self):
        bgSubThreshold = 10
        k = cv2.waitKey(10)
        if k == ord('b'):  # presionar b para capturar el background
            self.bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            time.sleep(2)
            self.isBgCaptured = 1
            print('Background captured')

        elif k == ord('r'):  # presionar r para resetear el background
            time.sleep(1)
            self.bgModel = None
            self.triggerSwitch = False
            self.isBgCaptured = 0
            print('Reset background')

    # función que remueve el background
    def remove_background(self, frame):
        learningRate = 0
        fgmask = self.bgModel.apply(frame, learningRate = learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    # método con la que se procesa la imagen
    def process(self):
        # Cámara
        camera = cv2.VideoCapture(0)

        cap_region_x_begin = 0.5  # start point/total width
        cap_region_y_end = 0.8  # start point/total width
        threshold = 60  # umbral binario
        blurValue = 41  # parámetro de desenfoque gaussiano

        while camera.isOpened():
            ret, frame = camera.read()
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # filtro suavizador
            frame = cv2.flip(frame, 1)  # voltea el fotograma horizontalmente
            cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                          (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

            cv2.imshow('original', frame)

            if self.isBgCaptured == 0:
                self.set_background()
            # Corre una vez capturado el background
            elif self.isBgCaptured == 1:
                img = self.remove_background(frame)
                img = img[0:int(cap_region_y_end * frame.shape[0]),
                      int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # recorta la región de interés

                # Convierte la imagen en una binaria
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
                ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Add prediction and action text to thresholded image
                # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
                # Draw the text
                cv2.putText(thresh, "Prediction: {} ({}%)".format(prediction, score), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255))
                cv2.putText(thresh, "Action: {}".format(action), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255))  # Draw the text
                cv2.imshow('ori', thresh)

                # Obtiene los contornos
                thresh1 = copy.deepcopy(thresh)
                _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                length = len(contours)
                maxArea = -1
                if length > 0:
                    for i in range(length):  # Encuentra el mayor contorno (según el área)
                        temp = contours[i]
                        area = cv2.contourArea(temp)
                        if area > maxArea:
                            maxArea = area
                            ci = i

                    res = contours[ci]
                    hull = cv2.convexHull(res)
                    drawing = np.zeros(img.shape, np.uint8)
                    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                cv2.imshow('output', drawing)

                # se realiza la predicción
                self.predicion(thresh)
                cv2.waitKey(1)

                # se publica la acción asociada
                self.publisher.publish(self.twist)

        cap.release()

def main():
    rospy.init_node('test') #creacion y registro del nodo

    obj = Template('args') # Crea un objeto del tipo Template, cuya definicion se encuentra arriba

    obj.process()


if __name__ =='__main__':
	main()
