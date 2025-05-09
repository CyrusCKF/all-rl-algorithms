import logging

from rl_server.socket_connection import SocketConnection

logging.basicConfig(level=logging.DEBUG)

PORT = 11008
conn = SocketConnection(port=PORT)

conn.send("Hello from Python")
message = conn.receive()
print(f"Received from client `{message}`")
