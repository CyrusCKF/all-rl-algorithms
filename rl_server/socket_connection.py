import logging
import socket
from select import select

logger = logging.getLogger(__name__)


class SocketConnection:
    def __init__(self, address: str = "127.0.0.1", port: int = 11009) -> None:
        logger.info(f"waiting for remote connection on {address=} {port=}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.bind((address, port))
        sock.listen(0)
        sock.settimeout(5)
        try:
            self.connection, self.client_address = sock.accept()
            logger.info("connection established")
        except TimeoutError:
            self.connection = None
            logger.info("connection time out")

    def receive(self) -> str | None:
        """Receive data in a non-blocking way. Return None if no available bytes"""
        if self.connection is None:
            return None

        readable, writable, exceptional = select([self.connection], [], [], 0.01)
        if not readable:
            return None

        len_data = self.connection.recv(4)
        length = int.from_bytes(len_data, "little")
        data = self.connection.recv(length)
        return data.decode()

    def send(self, text: str):
        if self.connection is None:
            return
        message = len(text).to_bytes(4, "little") + bytes(text.encode())
        self.connection.sendall(message)
