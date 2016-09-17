##############################################################################
# FILE : ex12.py
# WRITER : HANIT_HAKIM 308396480, INBAR_wOLFF 204641708
# EXERCISE : intro2cs ex12 2015-2016
# DESCRIPTION : In this exercise we've created a drawing program
# which allows multiple users to connect to and draw together.
# Each user can choose a group he would like to draw in and join
# the other users in this group to draw on a joint canvas.
##############################################################################


##############################################################################
# Imports
##############################################################################
import socket
import gui
import sys
import select
import tkinter.messagebox
import tkinter as tk

##############################################################################
# Class Client
##############################################################################
BUFFER_SIZE = 1024
MSG_DELIMITER = '\n'
WAITING_FOR_MESSAGE_TIME = 0.2
NUMBER_OF_ARGUMENTS = 4
CORRECT_NAME_LENGTH = 20
ERROR_MESSAGE_WRONG_USER_NAME = "The user name you've inserted is invalid " \
                                "Please use only english letters or numbers."
ERROR_MESSAGE_WRONG_GROUP_NAME = "The group name you've inserted is invalid " \
                                 "Please use only english letters or numbers."


class Client:
    """
    A class representing a client in the drawing program.
    In the client class we create a client object for each user that logs to
    the drawing program.
    A user can join an existing group or create a new one. If he joins an
    existing group, his drawing interface will be updated with the shapes
    that the other users already drew.
    """

    def __init__(self, server_address, server_port, user_name, group_name):
        """
        Initializes an object client in the interface.
        :param parent: the root
        :param group_name: the group a user is in
        :param user_name: the user's name
        :param server_address: the address of the server we use for connecting
        :param server_port: the port in which we use to communicate with the
        server
        :param socket: includes the server_port and the server_ip
        """
        self.__parent = tk.Tk()
        self.__server_address = server_address
        self.__server_port = server_port
        self.__user_name = user_name
        self.__group_name = group_name
        self.__socket = socket.socket()
        self.__gui = gui.Gui(self.__parent, group_name, user_name)
        self.__parent.protocol("WM_DELETE_WINDOW", self.on_closing)

    def connect_and_inform_server(self):
        """
        The function connects the client to the server and send a
        connecting message to the server.
        """
        self.__socket.connect((self.__server_address, self.__server_port))
        connected_msg = 'join;' + self.__user_name + ';' + self.__group_name \
                        + '\n'
        connected_msg_in_bytes = bytes(connected_msg, 'ascii')
        self.__socket.sendall(connected_msg_in_bytes)

    def receive_server_messages(self):
        """
        The function receives the messages from the server and sorting them
        according to the message content.
        """
        # we want to check information for the socket to read
        is_readable = [self.__socket]
        is_writable = []
        is_error = []
        r, w, x = select.select(is_readable, is_writable, is_error,
                                WAITING_FOR_MESSAGE_TIME)
        for sock in r:
            if sock == self.__socket:
                initial_bytes_str = b''
                while initial_bytes_str is not None:
                    current_msg = self.__socket.recv(BUFFER_SIZE)
                    full_current_msg = initial_bytes_str + current_msg
                    decoded_msg = full_current_msg.decode('ascii')
                    while MSG_DELIMITER in decoded_msg:
                        relevant_msg = decoded_msg[
                                       :decoded_msg.index(MSG_DELIMITER)]
                        initial_bytes_str = bytes(decoded_msg[
                                                  decoded_msg.index(
                                                      MSG_DELIMITER) + 1:],
                                                  'ascii')
                        contents_of_msg = relevant_msg.split(';')
                        if 'join' in contents_of_msg:
                            # adding to the attribute users list the user name
                            self.__gui.update_user_lst_and_group(
                                contents_of_msg[1])
                        if 'shape' in contents_of_msg:
                            self.updating_shapes_in_gui(contents_of_msg)
                        if 'leave' in contents_of_msg:
                            self.__gui.set_remove_from_users_list(
                                contents_of_msg[1])
                            self.__gui.update_user_lst_and_group('')
                        if 'users' in contents_of_msg:
                            names_to_add = contents_of_msg[1].split(',')
                            for name in names_to_add:
                                self.__gui.update_user_lst_and_group(name)

                        decoded_msg = initial_bytes_str.decode("ascii")

                    if initial_bytes_str == b'':
                        initial_bytes_str = None

        # we are calling this function from within so we can use the loop
        # that is created by the after function
        self.shape_drawing_data_for_server()
        self.__parent.after(100, self.receive_server_messages)

    def shape_drawing_data_for_server(self):
        """
        The function organizes the data that receives after a user drew a
        shape, for the function shape_message_to_server.
        """
        if self.__gui.get_param_lst_for_server() != []:
            coordinates = []
            shape = self.__gui.get_param_lst_for_server()[0]
            for tuple in shape[0]:
                coordinates.append(tuple)
            the_shape = shape[1]
            color = shape[2]
            self.__gui.set_param_lst_for_server(shape)
            self.shape_message_to_server(the_shape, coordinates, color)

    def shape_message_to_server(self, the_shape, coordinates, color):
        """
        This function will be called from gui, in order to inform the server
        about a client's drawing.
        receives:
        the_shape = string with the name of the shape that the client drew
        coordinates = list of tuples by x,y - ordering the location of the
        shape.
        color = string of the color of the shape.
        """
        coordinates_for_msg = []
        for spot in coordinates:
            for cord in spot:
                coordinates_for_msg.append(cord)
        str_coords = ','.join(str(num) for num in coordinates_for_msg)
        drawing_msg_to_server = 'shape;' + the_shape + ';' + str(str_coords) \
                                + ';' + color + '\n'
        drawing_msg_to_server_in_bytes = bytes(drawing_msg_to_server, 'ascii')
        self.__socket.sendall(drawing_msg_to_server_in_bytes)

    def updating_shapes_in_gui(self, contents_of_msg):
        """
        This function will receive the contents of the relevant message,
        and will return a list of the relevant parameters. it will be used
        in gui in order to update the canvas.
        """
        name_of_user = contents_of_msg[1]
        shape_name = contents_of_msg[2]
        shape_color = contents_of_msg[4]
        # a string of numbers, by: x1,y1,x2,y2...
        shape_requested_spot = contents_of_msg[3].split(',')
        int_list = [int(x) for x in shape_requested_spot]
        shape_coord = []
        for i in range(len(int_list) // 2):
            tup = (int_list[0], int_list[1])
            shape_coord.append(tup)
            int_list.remove(int_list[0])
            int_list.remove(int_list[0])
        self.__gui.set_coord_lst()
        self.__gui.draw_shape(shape_name, shape_color, name_of_user,
                              shape_coord)

    def on_closing(self):
        """
        The function will be called from gui in the minute the user closes the
        graphic window
        """
        # send the leaving msg
        # disconnect socket
        leave_msg_to_server_in_bytes = bytes('leave', 'ascii')
        self.__socket.sendall(leave_msg_to_server_in_bytes)
        self.__socket.close()
        self.__parent.destroy()

    def start_app(self):
        """
        The function activates the receiving messages from the server and
        runs the gui's mainloop.
        """
        self.receive_server_messages()
        self.__gui.start_gui()


def run_client(server_address, server_port, user_name, group_name):
    """
    The main function which creates a client object, connecting it to the
    server and activates gui.
    """
    a_client = Client(server_address, server_port, user_name, group_name)
    a_client.connect_and_inform_server()
    a_client.start_app()


# writing sys for command line...

if __name__ == "__main__":
    if len(sys.argv) == NUMBER_OF_ARGUMENTS + 1:
        script_name = sys.argv[0]
        server_address = sys.argv[1]
        server_port = int(sys.argv[2])
        user_name = sys.argv[3]
        group_name = sys.argv[4]
        if not user_name.isalnum() or len(user_name) > CORRECT_NAME_LENGTH:
            tk.messagebox.showinfo("ERROR!", ERROR_MESSAGE_WRONG_USER_NAME)
        if not group_name.isalnum() or len(group_name) > CORRECT_NAME_LENGTH:
            tk.messagebox.showinfo("ERROR!", ERROR_MESSAGE_WRONG_GROUP_NAME)

        else:
            run_client(server_address, server_port, user_name, group_name)
