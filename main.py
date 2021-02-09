from kivy.app import App 
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import json, glob
from datetime import datetime
from pathlib import Path
import random
from hoverable import HoverBehavior
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import os
from kivy.uix.videoplayer import VideoPlayer
Builder.load_file('design.kv')

class LoginScreen(Screen):
    def sign_up(self):
        self.manager.current = "sign_up_screen"
    
    def login(self,uname,pword):
        with open("users.json") as file:
            users= json.load(file)
        if uname in users and users[uname]['password'] == pword:
            self.manager.current = 'login_screen_success'
        else:
            self.ids.login_wrong.text = "Wrong username or password!"

class MyVideoApp(Screen):

    def startvideo(self):
        print('entered startvideo')
        self.player= VideoPlayer(source='output2.mp4',  state='play', options={'allow_stretch': True})
        return (self.player)

     
class RootWidget(ScreenManager):
    pass



class MyWidget(Screen):
    
    def detect(frame, net, transform):
        height, width = frame.shape[:2]
        frame_t = transform(frame)[0]
        x = torch.from_numpy(frame_t).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)
        detections = y.data
        scale = torch.Tensor([width, height, width, height])
        # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame
    
    def start(self, filename):
        try:
            
            
            file1=filename[0]
            print(file1)
            self.ids.wait.text = "Video Processing....... This may take a while..."
            # Creating the SSD neural network
            net = build_ssd('test')
            net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
            transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
            reader = imageio.get_reader(file1)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer('output2.mp4', fps = fps)
            for i, frame in enumerate(reader):
                frame = MyWidget.detect(frame, net.eval(), transform)
                writer.append_data(frame)
                print(i)
            writer.close()
            print("completed successfully")
            print('video playing..')
            self.manager.current= "videoapp"
            
            
        except:
            print("error")
        




class Convert(Screen):
    def start_convert(self,filename):
        pass


class SignUpScreen(Screen):
    def add_user(self,uname,pword):
        with open("users.json") as file:
            users = json.load(file)
        users[uname]= { 'username': uname, 'password':pword ,'created': datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        }

        with open("users.json",'w') as file:
            json.dump(users,file)
        self.manager.current= "sign_up_screen_success"

class SignUpSceenSuccess(Screen):
    def go_to_login(self):
        self.manager.transition.direction='right'
        self.manager.current= "login_screen"


class LoginScreenSuccess(Screen):
    def log_out(self):
        self.manager.transition.direction = "right"
        self.manager.current = "login_screen"

    def got_to_file_chooser(self):
        self.manager.current = 'choose_file'
    
    def get_quote(self,feel):
       
        feel = feel.lower()
        available_feelings = glob.glob("quotes/*txt")
        

        available_feelings = [Path(filename).stem for filename in available_feelings]
        
        if feel in available_feelings:
            filename=f"quotes/{feel}.txt"
            with open(filename,encoding='utf-8') as file:
                quotes = file.readlines()
            self.ids.quote.text = random.choice(quotes)

class ImagesButton(ButtonBehavior,HoverBehavior, Image ):
    pass

class FinalVideo(Screen):
    pass

class MainApp(App):
    def build(self):
        return RootWidget()

if __name__=="__main__": 
    MainApp().run()