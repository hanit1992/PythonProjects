##############################################################################
# Imports
##############################################################################
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox


##############################################################################
# Class Gui
##############################################################################
HEIGHT = 500
WIDTH = 500
HELP_MESSAGE = "welcome to the ultimate drawing board!\n if you are " \
               "a bad drawer - this program is perfect for you!\n\n" \
               "About The Program:\n" \
               "This program will allow you to paint with other " \
               "users on a shared painting canvas.\n " \
               "Each user can choose from the interface's options " \
               "shapes and colors and draw on the canvas.\n" \
               "When the user draws a shape, his name will appear on the " \
               "shape's side.\n" \
               "In this way you " \
               "can be updated on the shapes each user in your group draws. " \
               "\n\n" \
               "Interface Options:\n" \
               "Canvas - on the bottom right of the interface you can " \
               "see the drawing canvas on which you can use to paint " \
               "different shapes in different colors.\n" \
               "List box - on the right interface you can see the list " \
               "box which displays the group you are in and the other " \
               "online users in this group.\n" \
               "Shapes buttons - above the canvas is located the " \
               "shapes toolbar, that can be used to draw different " \
               "shapes on the canvas.\n" \
               "Color buttons - on the right side of the canvas is " \
               "located the colors toolbar from which you can use to " \
               "fill the shape you'll choose.\n" \
               "\n\n" \
               "Enjoy!\n"


class Gui:
    """
    A class representing a gui in the drawing program.
    In the gui class we create an interface for the user in which he can use
    to draw with other users.
    """

    def __init__(self, parent, group, user_name):
        """
        Initializes an object gui in the interface.
        :param parent: the root
        :param group: the group a user is in
        :param user_name: the user's name
        :param canvas: the drawing board attribute
        :param current_color: the color the user chose
        :param current_shape: the shape the user chose
        """
        self.__parent = parent
        self.__parent.resizable(False, False)
        self.__group = group
        self.__user_name = user_name
        self.__canvas = None
        self.__current_color = 'black'  # default color
        self.__current_shape = 'line'  # default shape
        self.__a_shape_was_drawn = False
        self.__listbox = ''
        self.__users_lst = []
        self.__coord_lst = []
        self.__param_lst_for_server = []
        self.__p_1_shapes = tk.PhotoImage(file='triangle.gif')
        self.__p_2_shapes = tk.PhotoImage(file='rectangle.gif')
        self.__p_3_shapes = tk.PhotoImage(file='oval.gif')
        self.__p_4_shapes = tk.PhotoImage(file='line.gif')
        self.__p_1_color = tk.PhotoImage(file='black.gif')
        self.__p_2_color = tk.PhotoImage(file='violet.gif')
        self.__p_3_color = tk.PhotoImage(file='blue.gif')
        self.__p_4_color = tk.PhotoImage(file='green.gif')
        self.__p_5_color = tk.PhotoImage(file='yellow.gif')
        self.__p_6_color = tk.PhotoImage(file='orange.gif')
        self.__p_7_color = tk.PhotoImage(file='red.gif')
        self.__header = ImageTk.PhotoImage(Image.open("header.jpg"))
        self.__logo = ImageTk.PhotoImage(Image.open("logo.jpg"))
        self.main_window()
        self.__canvas.bind("<Button-1>", self.get_coordinates)

    def start_gui(self):
        """
         The function returns mainloop for class Client to use.
        """
        self.__parent.mainloop()

    def get_users_list(self):
        """
        The function returns the users list for Client class to use.
        """
        return self.__users_lst

    def set_remove_from_users_list(self, name):
        """
        The function returns the users list so Client class can update it.
        """
        self.__users_lst.remove(name)

    def get_param_lst_for_server(self):
        """
        The function returns the parameters list for class Client to use.
        """
        return self.__param_lst_for_server

    def set_param_lst_for_server(self, value):
        """
        The function returns the parameters list so class Client can update it.
        """
        if value == []:
            self.__param_lst_for_server = []
        else:
            self.__param_lst_for_server.remove(value)

    def set_coord_lst(self):
        """
        The function returns the coordinates list so class Client can change
        it.
        """
        self.__coord_lst = []

    def canvas_location(self, lower_frame):
        """
        The function defines the boarders of canvas in the Gui.
        """
        canvas = tk.Canvas(lower_frame, width=WIDTH, height=HEIGHT,
                           bg='white')
        canvas.grid(row=0, column=10, sticky=tk.E)
        self.__canvas = canvas

    def main_window(self):
        """
        This function includes commands and functions which places and
        displays the base objects on the gui. The base objects includes:
        the main frames, the header, the logo, the group and users list box,
        the colors button, the shapes buttons and the canvas location.
        """
        upper_frame = tk.Frame(self.__parent)
        upper_frame.grid(row=0, column=0)
        self.help_button(upper_frame)
        header = Image.open("header.jpg")
        self.__header = ImageTk.PhotoImage(header)
        header_label = tk.Label(upper_frame, image=self.__header)
        header_label.grid(row=1, column=2)
        logo = Image.open("logo.jpg")
        self.__logo = ImageTk.PhotoImage(logo)
        logo_print = tk.Label(upper_frame, image=self.__logo)
        logo_print.grid(row=1, column=3)
        self.shape_buttons(upper_frame)
        separator = tk.Label(self.__parent)
        separator.grid(row=1, column=0)
        lower_frame = tk.Frame(self.__parent)
        lower_frame.grid(row=2, column=0, sticky=tk.E, ipadx=5)
        self.group_and_users_list_box(lower_frame)
        separator = tk.Label(lower_frame, width=3)
        separator.grid(row=0, column=3)
        self.button_color(lower_frame)
        separator = tk.Label(lower_frame, width=3)
        separator.grid(row=0, column=8)
        self.canvas_location(lower_frame)

    def group_and_users_list_box(self, lower_frame):
        """
        The function defines the location of the users and group window as
        well as the scrollbar.
        """
        scrollbar = tk.Scrollbar(lower_frame)
        scrollbar.grid(row=0, column=2, ipady=225)
        self.__listbox = tk.Listbox(lower_frame, bg='white',
                                    yscrollcommand=scrollbar, height=31,
                                    width=25)

    def update_user_lst_and_group(self, username):
        """
        The function updates from the users list and adding new users to the
        window as new users logging. The function also updates the group
        that the user chose.
        """
        self.__listbox.delete(0, tk.END)
        if username != '':
            self.__users_lst.append(username)
        index = 1
        self.__listbox.insert(index, self.__group)
        for name in self.__users_lst:
            self.__listbox.insert(index, name)
            index += 2
        self.__listbox.grid(row=0, column=1)

    def choose_current_shape(self, shape):
        """
        The function updates the current shape that has been chosen by the
        user and returns a function which includes the relevant shape.
        """

        def return_func():
            if shape == 'rectangle':
                self.__current_shape = 'rectangle'
                self.__coord_lst = []
            if shape == 'triangle':
                self.__current_shape = 'triangle'
                self.__coord_lst = []
            if shape == 'line':
                self.__current_shape = 'line'
                self.__coord_lst = []
            if shape == 'oval':
                self.__current_shape = 'oval'
                self.__coord_lst = []

        return return_func

    def choose_current_color(self, color):
        """
        The function updates the current color that has been chosen by the
        user and returns a function which includes the relevant color.
        """

        def return_func():
            if color == 'black':
                self.__current_color = 'black'
                self.__coord_lst = []
            if color == 'blue':
                self.__current_color = 'blue'
                self.__coord_lst = []
            if color == 'green':
                self.__current_color = 'green'
                self.__coord_lst = []
            if color == 'yellow':
                self.__current_color = 'yellow'
                self.__coord_lst = []
            if color == 'red':
                self.__current_color = 'red'
                self.__coord_lst = []
            if color == 'orange':
                self.__current_color = 'orange'
                self.__coord_lst = []
            if color == 'violet':
                self.__current_color = 'violet'
                self.__coord_lst = []

        return return_func

    def get_coordinates(self, event):
        """
        The function sends the data (we'll mention bellow) to inform the
        server that a user would like to draw a shape. When user is clicking
        on the canvas to create a shape, the function receives the shapes'
        coordinates and insert it to a list. The function check how many
        coordinates are in the list, if there are two coordinates the function
        draw one of the two coordinates shapes (line\ oval\ rectangle).
        if there are three coordinates in the list the function call
        draw_shape to draw triangle.
        """
        self.__coord_lst.append((event.x, event.y))

        if self.__current_shape is 'line' and len(self.__coord_lst) == 2:
            self.parameters_for_server()
            self.__coord_lst = []
        if self.__current_shape is 'rectangle' and len(self.__coord_lst) == 2:
            self.parameters_for_server()
            self.__coord_lst = []
        if self.__current_shape is 'oval' and len(self.__coord_lst) == 2:
            self.parameters_for_server()
            self.__coord_lst = []
        if self.__current_shape is 'triangle' and len(self.__coord_lst) == 3:
            self.parameters_for_server()
            self.__coord_lst = []

    def draw_shape(self, shape, color, user_name, shape_coord):
        """
        This function draws the shapes on the canvas according to the user
        request.
        :param
        shape - the shape the user choose
        color - the color the user choose
        user_name
        """
        if shape == 'rectangle':
            self.__canvas.create_rectangle(shape_coord[0][0],
                                           shape_coord[
                                               0][1], shape_coord[1][0],
                                           shape_coord[1][1],
                                           fill=color, outline=color)
        if shape == 'line':
            self.__canvas.create_line(shape_coord[0][0], shape_coord[
                0][1], shape_coord[1][0], shape_coord[1][1],
                                      fill=color)
        if shape == 'oval':
            self.__canvas.create_oval(shape_coord[0][0], shape_coord[
                0][1], shape_coord[1][0], shape_coord[1][1],
                                      fill=color, outline=color)
        if shape == 'triangle':
            self.__canvas.create_polygon(shape_coord[0][0],
                                         shape_coord[
                                             0][1], shape_coord[1][0],
                                         shape_coord[1][1],
                                         shape_coord[2][0],
                                         shape_coord[2][1],
                                         fill=color, outline=color)
        self.__canvas.create_text(shape_coord[0][0], shape_coord[0][
            1], text=user_name)

    def parameters_for_server(self):
        """
        The function gathers the following parameters: coordinates, current
        shape and current color, that are needed for the
        communication with the server and inserts it to a dictionary.
        The dictionary is updating after each request from a user to
        draw a shape.
        """
        self.__param_lst_for_server.append((self.__coord_lst,
                                            self.__current_shape,
                                            self.__current_color))

    def shape_buttons(self, upper_frame):
        """
        The function creates the shape buttons on the canvas.
        """
        shapes_buttons_frame = tk.Frame(upper_frame)
        shapes_buttons_frame.grid(row=3, column=3)
        self.__p_1_shapes = tk.PhotoImage(file='triangle.gif')
        triangle_button = tk.Button(shapes_buttons_frame,
                                    image=self.__p_1_shapes,
                                    command=self.choose_current_shape(
                                        'triangle'))
        triangle_button.grid(row=0, column=1)
        self.__p_2_shapes = tk.PhotoImage(file='rectangle.gif')
        rectangle_button = tk.Button(shapes_buttons_frame,
                                     image=self.__p_2_shapes,
                                     command=self.choose_current_shape(
                                         'rectangle'))
        rectangle_button.grid(row=0, column=2)
        self.__p_3_shapes = tk.PhotoImage(file='oval.gif')
        oval_button = tk.Button(shapes_buttons_frame,
                                image=self.__p_3_shapes,
                                command=self.choose_current_shape(
                                    'oval'))
        oval_button.grid(row=0, column=3)
        self.__p_4_shapes = tk.PhotoImage(file='line.gif')
        line = tk.Button(shapes_buttons_frame,
                         image=self.__p_4_shapes,
                         command=self.choose_current_shape(
                             'line'))
        line.grid(row=0, column=4)

    def button_color(self, lower_frame):
        """
        The function creates the color buttons on the canvas.
        """
        shapes_buttons_frame = tk.Frame(lower_frame)
        shapes_buttons_frame.grid(row=0, column=7)
        self.__p_1_color = tk.PhotoImage(file='black.gif')
        black_button = tk.Button(shapes_buttons_frame,
                                 image=self.__p_1_color,
                                 command=self.choose_current_color('black'))
        black_button.grid(row=1, column=0)
        self.__p_2_color = tk.PhotoImage(file='violet.gif')
        violet_button = tk.Button(shapes_buttons_frame,
                                  image=self.__p_2_color,
                                  command=self.choose_current_color('violet'))
        violet_button.grid(row=2, column=0)
        self.__p_3_color = tk.PhotoImage(file='blue.gif')
        blue_button = tk.Button(shapes_buttons_frame,
                                image=self.__p_3_color,
                                command=self.choose_current_color('blue'))
        blue_button.grid(row=3, column=0)
        self.__p_4_color = tk.PhotoImage(file='green.gif')
        green_button = tk.Button(shapes_buttons_frame,
                                 image=self.__p_4_color,
                                 command=self.choose_current_color('green'))
        green_button.grid(row=4, column=0)
        self.__p_5_color = tk.PhotoImage(file='yellow.gif')
        yellow_button = tk.Button(shapes_buttons_frame,
                                  image=self.__p_5_color,
                                  command=self.choose_current_color('yellow'))
        yellow_button.grid(row=5, column=0)
        self.__p_6_color = tk.PhotoImage(file='orange.gif')
        orange_button = tk.Button(shapes_buttons_frame,
                                  image=self.__p_6_color,
                                  command=self.choose_current_color('orange'))
        orange_button.grid(row=6, column=0)
        self.__p_7_color = tk.PhotoImage(file='red.gif')
        red_button = tk.Button(shapes_buttons_frame,
                               image=self.__p_7_color,
                               command=self.choose_current_color('red'))
        red_button.grid(row=7, column=0)

    def help_button(self, upper_frame):
        """
        The function defines the boarders of the help button in the Gui.
        """
        help_button = tk.Button(upper_frame, text='Help',
                                command=self.help_button_msg)
        help_button.grid(row=0, column=0)

    def help_button_msg(self):
        """
        The function binds the help button to the message that will pop up
        when user presses it.
        """
        tk.messagebox.showinfo("Help!",
                               HELP_MESSAGE)
