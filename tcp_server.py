import socket

PORT = 11008
print(f"waiting for remote GODOT connection on port {PORT}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind the socket to the port, "localhost" was not working on windows VM, had to use the IP
server_address = ("127.0.0.1", PORT)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(0)
# sock.settimeout(60)
connection, client_address = sock.accept()
# connection.settimeout(GodotEnv.DEFAULT_TIMEOUT)
#        connection.setblocking(False) TODO
print("connection established")

string = "Hello from Python"
message = len(string).to_bytes(4, "little") + bytes(string.encode())
connection.sendall(message)

len_data = connection.recv(4)
length = int.from_bytes(len_data, "little")
data = connection.recv(length)
print(f"Received from client `{data.decode()}`")  
