import logging
import select

from rl_server.socket_connection import SocketConnection

logging.basicConfig(level=logging.DEBUG)

PORT = 11008
conn = SocketConnection(port=PORT)

conn.send("Hello from Python")

# message = conn.receive()
# print(f"Received from client `{message}`")

# readable, writable, exceptional = select.select([conn.connection], [], [], 0.01)
# print(readable, writable, exceptional)
# if readable:
#     # There is data to read
#     data = conn.receive()
#     print("Received data:", data)
# else:
#     print("No data available to read.")


# readable, writable, exceptional = select.select([conn.connection], [], [], 0.01)
# print(readable, writable, exceptional)
# if readable:
#     # There is data to read
#     data = conn.receive()
#     print("Received data:", data)
# else:
#     print("No data available to read.")

message = conn.receive()
print(f"Received from client `{message}`")
message = conn.receive()
print(f"Received from client `{message}`")
message = conn.receive()
print(f"Received from client `{message}`")
