import socket
s = socket.socket()
try:
    s.bind(("", 7860))
    print("Port is free")
    s.close()
except:
    print("Port is in use")