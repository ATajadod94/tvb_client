import kivy
kivy.require('1.10.1')
from kivy.app import App
from kivy.uix.button import Label

class tempelate(App):
    def build(self):
        return Label()

if __name__=="__main__":
    tempelate().run()
