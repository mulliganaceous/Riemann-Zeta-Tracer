#!/home/user/.pyenv/shims/python
import socket
import struct
import matplotlib.pyplot as plt
import numpy as np

PORT = 8081
WIDTH = 256
HEIGHT = 24000
COMPLEX_SIZE = 16
BATCHES = 1000

def create_plot():
    plt.figure(figsize=(10, 8))
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.axis('equal')  # Equal aspect ratio for proper complex plane
    return plt

def render(plt, p_data):
    title="Complex Numbers on Complex Plane"
    complex_array = np.array(p_data)
    real_parts = np.real(complex_array)
    imag_parts = np.imag(complex_array)
    plt.title(f'{title} ({len(complex_array)} points)')
    plt.scatter(real_parts, imag_parts, alpha=0.04, s=0.5)
    plt.show()

def connect():
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('localhost', PORT))
    plot = create_plot()
    total_data = 0
    data = []
    sumdata = 0
    frame = bytearray()
    while sumdata < WIDTH*HEIGHT*COMPLEX_SIZE:
        chunk = clientsocket.recv(WIDTH*HEIGHT*COMPLEX_SIZE//BATCHES)
        frame += bytearray(chunk)
        sumdata += len(chunk)
        print("Recieved %s bytes (%d total)\n" % (len(chunk), len(frame))) 
        i = 0
    while len(frame) and (i<len(frame)/16):
        point_data = (struct.unpack('dd', frame[i*16:(i+1)*16]))
        data.append(complex(point_data[0], point_data[1]))
        i = i + 1
        
    render(plot, data)

def main():
    print("Connecting...")
    connect()

main()
