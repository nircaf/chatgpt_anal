

import pygame
import time
import random
import os
import multiprocessing
import socket

# create a function to send a socket
def send_socket(ip_address, port, message):
    # create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # connect to the specified IP address and port
    s.connect((ip_address, port))

    # send the message
    s.sendall(message.encode())

    # close the socket
    s.close()

# create a function to receive a socket
def receive_socket(port):
    # create a socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # bind the socket to the specified port
    s.bind(("", port))

    # listen for incoming connections
    s.listen()

    # accept the incoming connection
    conn, addr = s.accept()

    # receive the message
    message = conn.recv(1024).decode()

    # close the socket
    s.close()

    # return the message
    return message

def main_game(port):
    # initialize Pygame
    pygame.init()

    # set the window size and create a window
    window_size = (400, 400)
    screen = pygame.display.set_mode(window_size)

    # create a serial connection to the port
    # replace "/dev/ttyUSB0" with the path to your port
    # ser = serial.Serial("/dev/ttyUSB0", 9600)

    # set the initial screen color to black
    color = (0, 0, 0)

    # run the game loop
    while True:
        # get the input from the serial port
        input = receive_socket(port)
        # input = input.decode("utf-8").strip() # decode the input and remove any leading or trailing whitespace

        # try to convert the input to a number between 1 and 10
        try:
            num = int(input)
            if num < 1 or num > 10:
                raise ValueError
        except ValueError:
            num = 0 # set the number to 0 if the input is not a valid number between 1 and 10

        # calculate the color based on the number
        r = int(255 * (num / 10)) # red value will range from 0 to 255
        g = int(255 * ((5 - abs(num - 5)) / 5)) # green value will range from 255 to 0 and back to 255
        b = int(255 * ((10 - num) / 10)) # blue value will range from 0 to 255
        color = (r, g, b)

        # fill the screen with the new color
        screen.fill(color)
        pygame.display.flip()

        # wait for 1 second before getting the next input
        time.sleep(1)

def send_number_to_port(port):
    # generate a random number between 1 and 10
    num = str(random.randint(1, 10))
    # get the hostname
    hostname = socket.gethostname()

    # get the IP address
    ip_address = socket.gethostbyname(hostname)
    # send the number to the port as a string
    send_socket(ip_address, port, num)


if __name__ == "__main__":
    port = 8000
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # get main_game running in a separate process
    p = multiprocessing.Process(target=main_game, args=(port,))
    p.start()
    while True:
        send_number_to_port(port)
        time.sleep(1)
