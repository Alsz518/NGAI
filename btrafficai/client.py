import json
import uuid
import logging
import functools

import zmq

class RPCError(Exception):
    def __init__(self, code=500, error='internal error'):
        self.code = code
        self.error = error


class BaseChatBotClient:
    logger = logging.getLogger(__name__)

    def __init__(self, server_url, client_id=None, timeout=120):
        self.server_url = server_url
        self.client_id = client_id or uuid.uuid4().hex
        self.timeout = timeout * 1000
        self.context = zmq.Context.instance()
        self.rpc_socket = self.context.socket(zmq.DEALER)
        self.rpc_socket.setsockopt(zmq.IDENTITY, self.client_id.encode("utf-8"))
        self.rpc_socket.connect(self.server_url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rpc_socket.close()

    def close(self):
        self.rpc_socket.close()

    @staticmethod
    def serialize(data):
        return json.dumps(data).encode("utf-8")

    @staticmethod
    def deserialize(data):
        return json.loads(data.decode("utf-8"))

    def _recv(self, method=None):
        try:
            return self.rpc_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            if not self.rpc_socket.poll(self.timeout, zmq.POLLIN):
                raise TimeoutError(method)
            return self.rpc_socket.recv_multipart()

    def connect(self):
        self.rpc_socket.send_multipart((b'', b':connect',))
        self._recv("connect")
        self.logger.debug("connected")

    def zmq_rpc(self, method, **kwargs):
        rpc_id = uuid.uuid4().hex.encode('utf-8')
        self.logger.debug("RPC request: %s %s", method, kwargs)
        kwargs['cmd'] = method
        self.logger.debug("RPC send_multipart")
        self.rpc_socket.send_multipart((b'', rpc_id, self.serialize(kwargs)))
        received = self._recv(method)
        self.logger.debug("RPC recv: %s", received)
        empty, recv_id, response = received
        result = self.deserialize(response)
        self.logger.debug("RPC response: %s", result)
        if result['code'] != 200:
            raise RPCError(result['code'], result['error'])
        return result['result']

    def __getattr__(self, item):
        if item.startswith('cmd_'):
            return functools.partial(self.zmq_rpc, method=item[4:])
        raise AttributeError(item)

    def start(self):
        self.connect()
        return self.zmq_rpc("start", client_id=self.client_id)
