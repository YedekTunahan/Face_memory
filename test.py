# Import kivy dependencies first
import threading

import tf as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Import other dependencies
import cv2
#import tensorflow as tf
import os
import numpy as np

from deepface.basemodels import Facenet512
from deepface.commons import functions
# import tensorflow as tf
import os

# Import other dependencies
import cv2
import numpy as np
from deepface.basemodels import Facenet512
from deepface.commons import functions
from kivy.app import App
# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.label import Label
from functools import partial
import json

from deepface import DeepFace
import atexit

from main import CamApp


# Build app and layout

class CamApp(App):


        happyValue = 0
        happyMoreValue = 0
        i = 0;
        k = 0;
        update_event = None


        print("İ deperi:",i)
        def build(self):
            # Main layout components
            print("BUİLD funciton çalıştı")
            self.web_cam = Image(size_hint=(1, .8))   # Ekran çerçevesi
            self.button = Button(text="Smile", on_press=self.verifySmile, size_hint=(.4, .1), pos_hint= {'center_x':.5, 'center_y': .5})  # Ekrandaki buttonumuz.on_pressmile basılınca çalışcak func belirtilmiş
            self.verification_label = Label(text="Look at the camera; make sure your face is in the yellow box.Smile", size_hint=(1, .1))    # EKRANIN ALTINDAKİ ALT AÇIKLAMA

            self.button.bind(on_press=partial(self.update_ui, buttonText="Smile More", labelText="Please Smile More "))

            # Add items to layout
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(self.web_cam)
            layout.add_widget(self.button)
            layout.add_widget(self.verification_label)

            # Setup video capture device
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0 / 33.0)  # bu olmadan görüntü gelmiyor.Bu Açılan kemarada belirtilen func'ı çalıştır demek,Kivy penceresidir,
                                                            # belirli bir işlevi belirli bir zaman aralğında düzenli olarak çağırmak için kullanılır. İşlevin her çağrısı  1 saniyede 33 kez gerçekleşeceği anlamına gelir
                                                            # 30FPS'i temsil eder her saniye 30 kare yakalar


            return layout

        # Run continuously to get webcam feed
        def update(self, *args):


               # Read frame from opencv
               ret, frame = self.capture.read()
               frame = frame[120:120 + 350, 200:200 + 350, :]

               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               # read haacascade to detect faces in input image
               face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                                    "haarcascade_frontalface_default.xml")

               # detects faces in the input image
               faces = face_cascade.detectMultiScale(gray, 1.1, 2)

               # loop over all the detected faces
               for (x, y, w, h) in faces:
                   # To draw a rectangle around the detected face
                   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

               # Flip horizontall and convert image to texture
               buf = cv2.flip(frame,
                              0).tostring()  # görüntüyü belirtilen bir eksene yansıtır  0: yatay eksen etrafında yansıtıldıgını ifade eder bunuda bayt dizisine dönüştürür.
               img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
               img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
               self.web_cam.texture = img_texture  # Bu kivy uygulamasında görüntüyü hüncelleyerek canlı bir önizleme veya video akışı sağlar





        # Verification function to verify person
        def verifySmile(self, *args):

            print("veriftSmile Function Çalıştı")
            SAVE_PATH = os.path.join('input_image.jpg')
            ret, frame = self.capture.read()
            frame = frame[120:120 + 350, 200:200 + 350, :]


            cv2.imwrite(SAVE_PATH, frame)  # kayıt etmeyi sağlıyor...
            image = cv2.imread("input_image.jpg")
            height, width, _ = image.shape
            print(f"yükseklik: {height} , genişlik: {width}")


            # Set verification text


        def update_ui(self, *args,buttonText,labelText):
            print("update_ui Function Çalıştı")
            # Update button text
            self.button.text = buttonText

            # Schedule an asynchronous update for verification label text
            Clock.schedule_once(partial(self.update_verification_label, labelText=labelText), 0)

            if self.i == 0:
                print("first if:")
                # Bind button press event to update_ui method
                self.button.bind(on_press=partial(self.update_ui, buttonText="Smile More", labelText="Please Smile More "))
                self.i += 1

            elif self.i != 0:
                print("else if")
    # Kivy'de, UI bileşenlerini güncellemek için asenkron işlemler kullanmanız gerekmektedir. Önerim, Clock.schedule_once yöntemini kullanarak bir işlevi asenkron olarak çağırmaktır.
        def update_verification_label(self, *args,labelText):
            print("update_verification_label Function Çalıştı")
            # Update verification label text
            self.verification_label.text = labelText

        def expression_of_emotion(self):
            try:
                objs = DeepFace.analyze(img_path="input_image.jpg", detector_backend="opencv")
                # objs_json = json.dumps(objs)
                print("type: ", type(objs))
                # print("type2:", type(objs_json))

                print("---------- analiz: ", objs[0])
                print("---------- analiz: ", objs[0]["age"])
            except Exception as error:
                print("İFADE ALGILAMADA HATA OLUŞTU",str(error))


 # atexit.register(build)
 #        app = CamApp()
 #        app.self.expression_of_emotion()
 #        print("xxx")

if __name__ == '__main__':
    CamApp().run()