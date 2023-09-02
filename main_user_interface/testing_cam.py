import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

class CameraApp(App):
    def build(self):
        camera = Camera(resolution=(640, 480), index=0)  # Adjust resolution and index as needed
        layout.add_widget(camera)


        layout = BoxLayout(orientation='vertical')
        label = Label(
            text="Click Here",
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            size_hint=(None, None)
        )
        layout.add_widget(label)
        return layout

if __name__ == '__main__':
    CameraApp().run()
