from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class SalesAssistant(App):
    def build(self):
        #returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.5, 0.6)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

        # top widget showing the name and purpose of app
        self.ttl_head = Label(
            text= "AI Sales Assistant",
            font_size= 35,
            color= '#00FFCE'
        )
        self.window.add_widget(self.ttl_head)

        # label widget
        # self.greeting = Label(
        #                 text= "What's your name?",
        #                 font_size= 18,
        #                 color= '#00FFCE'
        #                 )
        # self.window.add_widget(self.greeting)

        # text input widget
        # self.user = TextInput(
        #             multiline= False,
        #             padding_y= (20,20),
        #             size_hint= (1, 0.25)
        #             )

        #self.window.add_widget(self.user)

        # button widget
        self.button = Button(
                      text= "GET STARTED",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#00FFCE',
                      #remove darker overlay of background colour
                      # background_normal = ""
                      )
        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        return self.window

    def callback(self, instance):
        #making a new window when the button is pressed to move onto next screen
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.5, 0.6)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

# run Say Hello App Calss
if __name__ == "__main__":
    SalesAssistant().run()