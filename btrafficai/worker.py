#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration worker process.

[manager]
  ↓ ROUTER/DEALER: cmd/data
(main thread)
  ↓ REQ/REP: tasks (task_socket)
    ↑ PUSH/PULL: data (send_socket)
(task worker)
  |  ↑ REQ/REP: tasks
  |  (exact scheduler)
  ↓ Queue
(xcp async command tasks)
"""

import sys
import json
import time
import logging
import threading

import zmq

#from . import base_session
from . import chatbot_session



def serialize(data):
    return json.dumps(data).encode("utf-8")


def deserialize(data):
    return json.loads(data.decode("utf-8"))


TIMEOUTS = {
    'first_conn': 5,
    'started': 10,
    'idle': 300,
}


class UnauthorizedError(RuntimeError):
    pass


class ConnectionClosed(RuntimeError):
    pass


class WorkerClosing(Exception):
    def __init__(self, disconnected=False):
        self.disconnected = disconnected


class ValueInaccessible(RuntimeError):
    def __init__(self, results, errors):
        self.results = results
        self.errors = errors


class RPCError(RuntimeError):
    def __init__(self, code=500, error='internal error'):
        self.code = code
        self.error = error


class Worker:

    def __init__(self, name, url, kwargs_json):
        self.name = name
        self.url = url
        self.kwargs = json.loads(kwargs_json)
        self.logger = logging.getLogger('worker.%s' % name)
        self.started = time.monotonic()
        self.running = True
        self.context = zmq.Context.instance()
        self.main_tasks_socket = self.context.socket(zmq.REQ)
        self.main_tasks_socket.connect('inproc://task')
        self.last_receive_time = time.monotonic()
        self.socket_name = name.encode('utf-8')
        #self.user_session = base_session.BaseUserSession()
        self.user_session = chatbot_session.ChatBotSession()

    def dispatch(self, send_socket, cmd, **kwargs):
        self.logger.debug("dispatch %s(%s)", cmd, kwargs)
        try:
            fn = getattr(self, 'cmd_' + cmd)
        except AttributeError:
            raise RPCError(404, 'command not found: ' + cmd)
        return fn(send_socket, **kwargs)

    def cmd_start(self, send_socket, client_id, **kwargs):
        try:
            return self.user_session.start(client_id, **kwargs)
        except Exception:
            self.logger.exception('failed to connect for %s', client_id)
            raise ConnectionClosed

    def cmd_stop(self, send_socket):
        if self.user_session is None:
            raise ConnectionClosed
        result = self.user_session.stop()
        if result:
            raise ConnectionClosed
        return None

    def cmd_respond(self, send_socket, message):
        if self.user_session is None:
            raise RPCError(406, 'not started')
        return self.user_session.respond(message)

    def cmd_get_history(self, send_socket, full=False):
        return self.user_session.get_history(full)

    def cmd_clear_history(self, send_socket):
        return self.user_session.clear_history()

    def cmd_reset(self, send_socket):
        return self.user_session.reset()

    def tasks_process_msg(self, send_socket, msg):
        try:
            result = self.dispatch(send_socket, **msg)
        except RPCError as ex:
            return {'code': ex.code, 'error': ex.error}
        except ConnectionClosed:
            raise
        except TypeError as ex:
            self.logger.exception('failed to process message: %r', msg)
            return {'code': 406, 'error': str(ex)}
        except Exception as ex:
            self.logger.exception('failed to process message: %r', msg)
            return {'code': 500, 'error': '%s: %s' % (ex.__class__.__name__, ex)}
        return {'code': 200, 'result': result}

    def tasks_watchdog(self, send_socket):
        now = time.monotonic()
        if self.user_session is None:
            if self.started + TIMEOUTS['first_conn'] < now:
                self.logger.warning('timeout waiting for connection')
                send_socket.send(b':close')
                self.close()
            return
        if not self.user_session.started:
            if self.started + TIMEOUTS['started'] < now:
                self.logger.warning('timeout connecting')
                send_socket.send(b':close')
                self.close()
            return
        if self.last_receive_time + TIMEOUTS['idle'] < now:
            self.logger.warning('timeout waiting for packets (idle)')
            send_socket.send(b':close')
            self.close()

    def tasks_raw_message(self, send_socket, frames):
        self.last_receive_time = time.monotonic()
        req_id = frames[0]
        if frames[1] == b'':
            self.logger.debug("received server ping")
            send_socket.send_multipart((b':ping', req_id))
            return
        msg = deserialize(frames[1])
        self.logger.info('> %s', msg)
        try:
            result = self.tasks_process_msg(send_socket, msg)
        except ConnectionClosed:
            send_socket.send_multipart((b':cmd', req_id, serialize(
                {'code': 200, 'result': None})))
            send_socket.send_multipart((b':data', b''))
            send_socket.send(b':close')
            return
        # already except Exception
        self.logger.info('< %s', result)
        send_socket.send_multipart((b':cmd', req_id, serialize(result)))

    def thread_tasks(self):
        send_socket = self.context.socket(zmq.PUSH)
        send_socket.connect('inproc://send')
        task_server_socket = self.context.socket(zmq.REP)
        task_server_socket.bind('inproc://task')
        while self.running:
            frames = task_server_socket.recv_multipart()
            self.logger.debug("thread_tasks: %s", frames)
            if frames[0] == b':watchdog':
                self.tasks_watchdog(send_socket)
                task_server_socket.send(b'')  # scheduler delay
            #elif frames[0] == b'polling':
                #pass
            elif frames[0] == b':cmd':
                task_server_socket.send(b'')
                self.tasks_raw_message(send_socket, frames[1:])

    def thread_timer(self):
        task_server_socket = self.context.socket(zmq.REQ)
        task_server_socket.connect('inproc://task')
        #scheduler = ExactScheduler()
        #scheduler['watchdog'] = 1  # s
        while self.running:
            # the processing speed will block
            #task = scheduler.sleep()
            #if task == 'watchdog':
            time.sleep(10)
            task_server_socket.send(b':watchdog')
            task_server_socket.recv_multipart()

    def main_route_manager_msgs(self, manager_socket):
        frames = manager_socket.recv_multipart()
        self.logger.debug("manager_socket received: %s", frames)
        if frames[0] == b':close':
            raise WorkerClosing(disconnected=False)
        elif frames[0] == b':cmd':
            self.main_tasks_socket.send_multipart(frames)
            self.main_tasks_socket.recv_multipart()
        elif frames[0] == b':start':
            manager_socket.send(b':started')

    def start(self):
        manager_socket = self.context.socket(zmq.DEALER)
        manager_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        manager_socket.connect(self.url)
        # this socket belongs to the main thread
        inp_send_socket = self.context.socket(zmq.PULL)
        inp_send_socket.bind('inproc://send')

        thr_tasks = threading.Thread(
            target=self.thread_tasks, name='tasks', daemon=True)
        thr_tasks.start()
        thr_timer = threading.Thread(
            target=self.thread_timer, name='timer', daemon=True)
        thr_timer.start()

        # heartbeat
        manager_socket.send_multipart((b':connect', b''))

        poller = zmq.Poller()
        poller.register(manager_socket, zmq.POLLIN)
        poller.register(inp_send_socket, zmq.POLLIN)

        self.logger.info('worker started')
        while self.running:
            try:
                for socket, event in poller.poll(1000):
                    if socket == manager_socket:
                        self.main_route_manager_msgs(manager_socket)
                        continue
                    # inp_send_socket
                    frames = socket.recv_multipart(copy=False)
                    if frames[0].bytes == b':close':
                        raise WorkerClosing()
                    manager_socket.send_multipart(frames)
            except WorkerClosing:
                self.logger.debug('close signal received')
                break
        poller.unregister(inp_send_socket)
        poller.unregister(manager_socket)
        if self.user_session:
            try:
                self.user_session.stop()
            except Exception as ex:
                self.logger.warning('failed to disconnect: %s' % ex)
        self.close()
        inp_send_socket.close()
        manager_socket.close()
        self.logger.info('worker closed')
        return 0

    def close(self):
        self.running = False


if __name__ == '__main__':
    worker = Worker(*sys.argv[1:])
    sys.exit(worker.start())
