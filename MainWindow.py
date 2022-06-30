from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QWidget
from PyQt5.QtCore import (QThread,pyqtSignal)
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from cv2 import cvtColor
from sqlalchemy import false
from FaceRecognition import Face
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import *
from tensorflow import *
from PIL import Image

from keras.models import  load_model
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys
import cv2

class Ui_MainWindow(QWidget):
    filename = None
    url = None
    
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(772, 753)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.imageToFind = QtWidgets.QLabel(self.centralwidget)
        self.imageToFind.setText("")
        self.imageToFind.setPixmap(QtGui.QPixmap("Asset/yourimagehere.png"))
        self.imageToFind.setScaledContents(True)
        self.imageToFind.setAlignment(QtCore.Qt.AlignCenter)
        self.imageToFind.setWordWrap(False)
        self.imageToFind.setObjectName("imageToFind")
        self.gridLayout.addWidget(self.imageToFind, 0, 0, 2, 2)
        self.uploadButton = QtWidgets.QPushButton(self.centralwidget)
        self.uploadButton.setObjectName("uploadButton")
        self.gridLayout.addWidget(self.uploadButton, 2, 0, 1, 1)
        self.cameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.cameraButton.setObjectName("cameraButton")
        self.gridLayout.addWidget(self.cameraButton, 2, 1, 1, 1)
        self.detectButton = QtWidgets.QPushButton(self.centralwidget)
        self.detectButton.setEnabled(False)
        self.detectButton.setObjectName("detectButton")
        self.gridLayout.addWidget(self.detectButton, 2, 2, 1, 1)
        self.output = QtWidgets.QLabel(self.centralwidget)
        self.output.setObjectName("output")
        self.gridLayout.addWidget(self.output, 3, 0, 1, 2)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 4, 0, 1, 2)
        self.addNewPerson = QtWidgets.QPushButton(self.centralwidget)
        self.addNewPerson.setEnabled(False)
        self.addNewPerson.setObjectName("addNewPerson")
        self.gridLayout.addWidget(self.addNewPerson, 4, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 772, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        self.uploadButton.clicked.connect(self.openFileNameDialog)
        self.cameraButton.clicked.connect(self.startCamera)
        self.detectButton.clicked.connect(self.detectFace)
        self.addNewPerson.clicked.connect(self.confirmAdd)
        
        self.cameraWorker = CameraThread()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    
    # Tutup camera + camera worker thread
    def disableCamera(self):
        _translate = QtCore.QCoreApplication.translate
        if self.cameraWorker:
            if self.cameraWorker.ThreadActive:
                self.cameraWorker.stop()
                
        self.cameraButton.setText(_translate("MainWindow", "Open Camera"))
    

    # Upload file
    def openFileNameDialog(self):
        self.disableCamera()
        options = QFileDialog.Options() 
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.filename = fileName
        if fileName:
            print(fileName)
            imagePixmap = QtGui.QPixmap(fileName)
            imagePixmap = imagePixmap.scaled(800,600,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)
            self.imageToFind.setPixmap(imagePixmap)
            self.detectButton.setDisabled(False)
            self.url = fileName
        
    # Input orang/muka baru melalui nama di line edit
    def confirmAdd(self):
        listPeople = os.listdir("Data")
        personName = self.lineEdit.text()
        self.newFace.name = personName
        print(listPeople)
        print(personName)
        if not personName in listPeople:    
            self.createNewPerson() 
        else:
            self.saveFace()
        self.output.setText(personName + " has been Added!")
        self.lineEdit.setDisabled(True)
        self.lineEdit.setText("")
        self.addNewPerson.setDisabled(True)

    # Simpan muka yang telah dikenali ke database
    def saveFace(self):
        listFiles = os.listdir("Data/" + self.newFace.name)
        if(len(listFiles) <= 6):
            np.save("Data/"+ self.newFace.name+ "/" + self.newFace.name + str(len(listFiles) + 1),self.newFace.encoding)

    # Buat orang baru untuk disimpan dalam data jika input tidak ditemukan
    def createNewPerson(self):
        os.mkdir("Data/" + self.newFace.name)
        listFiles = os.listdir("Data/" + self.newFace.name)
        np.save("Data/"+ self.newFace.name+ "/" + self.newFace.name + str(len(listFiles) + 1),self.newFace.encoding)

    # Start camera + thread
    def startCamera(self):
        _translate = QtCore.QCoreApplication.translate
        if self.cameraWorker.ThreadActive:
            self.cameraWorker.saveImage()
            time.sleep(1)
            self.imageToFind.setPixmap(QPixmap("CameraResult/c1.png"))
            self.detectButton.setDisabled(False)
            self.cameraButton.setText(_translate("MainWindow", "Open Camera"))
            self.filename = "CameraResult/c1.png"
        else:
            self.cameraWorker.start()
            self.cameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)
            self.cameraButton.setText(_translate("MainWindow", "Take Photo"))
            self.detectButton.setDisabled(True)
       
    # Detecting muka menggunakan DetectWorker thread
    def detectFace(self):
        self.thread = DetectThread()
        self.thread.filename = self.filename
        self.thread.finished.connect(self.detectFinish)
        self.thread.start()
        
    # Output hasil detect
    def detectFinish(self):
        self.newFace = self.thread.newFace
        _translate = QtCore.QCoreApplication.translate
        if not self.thread.result:
            self.output.setText(_translate("MainWindow","Face not found, Insert another Image"))
        elif self.thread.name != "Unknown":
            self.output.setText(_translate("MainWindow","This is " + self.thread.name))
            self.saveFace()
        elif self.thread.name == "Unknown":
            self.output.setText(_translate("MainWindow","Person Unknown \nClick button below to add new person"))
            self.addNewPerson.setDisabled(False)
            self.lineEdit.setDisabled(False)
        
        self.detectButton.setDisabled(True)

    # UI
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition"))
        self.uploadButton.setText(_translate("MainWindow", "Choose File"))
        self.cameraButton.setText(_translate("MainWindow", "Open Camera"))
        self.detectButton.setText(_translate("MainWindow", "Detect Face"))
        self.output.setText(_translate("MainWindow", "Person Name"))
        self.addNewPerson.setText(_translate("MainWindow", "Add"))
    
    # UI
    def ImageUpdateSlot(self, Image):
        self.imageToFind.setPixmap(QPixmap.fromImage(Image))
        
    # Thread
    def CancelFeed(self):
        self.cameraWorker.stop()



class DetectThread(QThread):
    filename = ""
    name = "Unknown"
    result = False

    def run(self):
        self.newFace = Face("",self.filename)
        
        if self.newFace.result:
            self.result = True
            self.name = self.newFace.findPerson()
        else:
            self.result = False
        

class CameraThread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ThreadActive = False
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        
        # load model
        model = load_model("best_model.h5")
        face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while self.ThreadActive:
            ret, self.frame = Capture.read()           
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                
                
                faces_detected = face_haar_cascade.detectMultiScale(self.frame, 1.32, 5)
                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                    roi_gray = self.frame[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                    roi_gray = cv2.resize(roi_gray, (224, 224))
                    # img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
                    img_pixels = keras.utils.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255
                    
                                                    
                    predictions = model.predict(img_pixels)
                    
                    # find max indexed array
                    max_index = np.argmax(predictions[0])

                    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    predicted_emotion = emotions[max_index]

                    cv2.putText(self.frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("test")

                # resized_img = cv2.resize(self.frame, (1000, 700))
                # cv2.imshow('Facial emotion analysis ', resized_img)

                if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                    break
                    
                    
                # self.FlippedImage = cv2.flip(self.frame, 1)
                self.FlippedImage = self.frame
                ConvertToQtFormat = QImage(self.FlippedImage.data, self.FlippedImage.shape[1], self.FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                
    def saveImage(self):
        print("Image saved")
        image = cv2.cvtColor(self.frame, cv2.COLOR_RGBA2RGB)
        flippedImage = cv2.flip(image,1)
        cv2.imwrite('CameraResult/c1.png',flippedImage)
        # cv2.destroyAllWindows()
        self.stop()
                
    def stop(self):
        self.ThreadActive = False



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



