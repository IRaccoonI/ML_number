from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from ml import ML
from PIL import Image
import numpy as np

from time import time


from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)


class NumRes(Label):
    def __init__(self, num, **kwarg):
        super().__init__()
        self._num = num
        self.set(0)
        self.color = (0, 0, 0, 1)

    def set(self, val):
        self.text = f'{self._num}: {val}%'



class MyPaintWidget(Widget):
    def __init__(self, update_ev, **kwargs):
        super().__init__(**kwargs)
        self.update_ev = update_ev
        self.last_time_ev = 0

    def on_touch_down(self, touch):
        color = (0, 0, 0)
        with self.canvas:
            Color(*color, mode='hsv')
            d = 8.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=d)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

        if time() > self.last_time_ev + 1:
            self._update_ev()

    def on_touch_up(self, touch):
        self.update_ev()


    def _update_ev(self):
        self.last_time_ev = time()
        self.update_ev()

class MyPaintApp(App):

    def build(self):
        main_box = BoxLayout()

        painter_size = '500dp'
        draw_parent = BoxLayout(orientation='vertical',
            size_hint_x = None,
            width = painter_size
        )
        labels_box = BoxLayout(orientation='vertical')
        main_box.add_widget(draw_parent)
        main_box.add_widget(labels_box)

        self.painter = MyPaintWidget(self.update_res,
            size_hint_y = None,
            height = painter_size
        )
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)

        draw_parent.add_widget(self.painter)
        draw_parent.add_widget(clearbtn)

        # addbtn = Button(text='add')
        # addbtn.bind(on_release=self.add_ev)
        # labels_box.add_widget(addbtn)
        w_nums = []
        for i in range(10):
            n_w = NumRes(i)
            w_nums.append(NumRes(i))
            labels_box.add_widget(n_w)

        self.w_nums = w_nums
        self.labels_box = labels_box


        self.machine = ML((28*28, 16, 16, 10, ))
        self.machine.load_weight('weights_m.out')

        return main_box

    def clear_canvas(self, obj):
        self.painter.canvas.clear()


    def update_res(self):
        self.painter.export_to_png('test.png')
        image = Image.open('test.png')
        im = np.asarray(image)[..., 3] / 255
        f = True
        while f:
            f = False
            if min(im.shape) > 28 and np.sum(im[:, 0]) == 0 and np.sum(im[0, :]) == 0:
                # remove first row
                im = im[1:, :]
                # remove first col
                im = im[:, 1:]
                f = True
            if min(im.shape) > 28 and np.sum(im[:, -1]) == 0 and np.sum(im[-1, :]) == 0:
                # remove last row
                im = im[:-1, :]
                # remove last col
                im = im[:, :-1]
                f = True
            if min(im.shape) > 28 and np.sum(im[:, -1]) == 0 and np.sum(im[0, :]) == 0:
                # remove first row
                im = im[1:, :]
                # remove last col
                im = im[:, :-1]
                f = True
            if min(im.shape) > 28 and np.sum(im[:, 0]) == 0 and np.sum(im[-1, :]) == 0:
                # remove last row
                im = im[:-1, :]
                # remove first col
                im = im[:, 1:]
                f = True
        image = Image.fromarray(np.uint8(im*255)).resize((20, 20, ))
        im = np.asarray(image) / 255

        for i in range(4):
            im = np.insert(im, 0, np.zeros((20)), axis =1)
        for i in range(4):
            im = np.append(im, np.zeros((20, 1)), axis =1)
        for i in range(4):
            im = np.insert(im, 0, np.zeros((28)), axis =0)
        for i in range(4):
            im = np.append(im, np.zeros((1, 28,)), axis =0)

        image = Image.fromarray(np.uint8(im*255))
        image.save('test.png')


        res = self.machine(im.squeeze().reshape(28*28))[-1]
        res_dict = {
            k: res[k] for k in range(len(res))
        }
        res_dict = {
            k: int(round(v * 100)) for k, v in res_dict.items()
        }
        res_tup = tuple(sorted(res_dict.items(), key=lambda item: item[1]))
        # print(res_dict)
        # print(res)
        self.labels_box.clear_widgets()
        for n, v in reversed(res_tup):
            self.w_nums[n].set(v)
            self.labels_box.add_widget(self.w_nums[n])

        # self.painter.canvas.clear()




if __name__ == '__main__':
    MyPaintApp().run()