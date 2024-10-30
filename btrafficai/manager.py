#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import logging
import weakref
import threading
import contextlib
import subprocess
import collections.abc

import zmq


logger = logging.getLogger('session_manager')


def serialize(data):
    return json.dumps(data).encode("utf-8")


def deserialize(data):
    return json.loads(data.decode("utf-8"))


class KeyedLocks(collections.abc.Collection):
    """
    Lock with names.
    Automatically create and delete locks for specified names.
    """
    def __init__(self):
        self.locks = weakref.WeakValueDictionary()

    def __len__(self) -> int:
        return len(self.locks)

    def __getitem__(self, item) -> 'threading.RLock':
        lock = self.locks.get(item)
        if lock is None:
            self.locks[item] = lock = threading.RLock()
        return lock

    def __delitem__(self, key) -> None:
        try:
            del self.locks[key]
        except KeyError:
            pass

    def __iter__(self):
        return iter(self.locks)

    def __contains__(self, item):
        return item in self.locks

    def keys(self):
        return self.locks.keys()

    def items(self):
        return self.locks.items()


class WorkerProcess:
    kill_wait = 5

    def __init__(self, context, name, backend_url, script_name, environ, **kwargs):
        self.context = context
        self.name = name
        self.name_str = name.decode('utf-8', errors='replace')
        self.backend_url = backend_url
        self.script_name = script_name
        self.environ = environ or {}
        self.kwargs = kwargs
        self.proc = None
        self.started = threading.Event()
        self.closed = False
        self.last_activity = time.monotonic()

    def start(self):
        environ = os.environ.copy()
        environ.update(self.environ)
        self.proc = subprocess.Popen((
            sys.executable, self.script_name, self.name_str, self.backend_url,
            json.dumps(self.kwargs)
        ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environ)
        started = self.started.wait(5)
        if not started:
            raise TimeoutError('starting process timeout: %s' % self.name_str)
        logger.info('started process %s', self.name_str)

    def keep_alive(self):
        self.last_activity = time.monotonic()

    def send(self, req_id, payload):
        try:
            packet_len = len(req_id) + 1 + len(payload)
            self.proc.stdin.write(packet_len.to_bytes(4, 'little'))
            self.proc.stdin.write(req_id + b'\0')
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (ValueError, OSError, BrokenPipeError):
            logger.warning("can't send message to %s: %.120s", self.name, payload)
            self.close()
        except Exception:
            logger.exception("can't send message to %s: %.120s", self.name, payload)

    def recv(self):
        ln = self.proc.stdout.readline()
        return deserialize(ln)

    def check(self, timeout):
        if self.proc.poll() is not None:
            return 'exit'
        if self.last_activity + timeout < time.monotonic():
            return 'timeout'
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        logger.info('worker process closing %s' % self.name)
        with contextlib.suppress(OSError):
            self.proc.stdin.close()
        try:
            return self.proc.wait(self.kill_wait)
        except subprocess.TimeoutExpired:
            pass
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        except OSError:
            logger.exception("can't kill process %s", self.name)
        logger.warning('killed %s' % self.name)
        return self.proc.returncode or -9


class SessionManager:
    """
    User Session Connection Manager.

    backend   REQ    <-- server_socket  --> ROUTER manager
      name, '', cmd --> manager --stdin--> worker

    worker    DEALER <-- backend_socket --> ROUTER manager
      'cmd'/'xcp'/'daq'/''
      'xcp', xcp1, ...
    """
    logger = logging.getLogger(__name__)
    heartbeat = 10
    activity_timeout = 300

    def __init__(self, server_url, processor_path, environ=None):
        self.running = True
        self.server_url = server_url

        self.context = zmq.Context.instance()
        self.poller = zmq.Poller()

        self.server_socket = self.context.socket(zmq.ROUTER)
        self.server_socket.bind(server_url)

        self.backend_socket = self.context.socket(zmq.ROUTER)
        port = self.backend_socket.bind_to_random_port('tcp://127.0.0.1')
        self.backend_url = 'tcp://127.0.0.1:%s' % port
        self.delayed_msg_socket = self.context.socket(zmq.PAIR)
        self.delayed_msg_socket.bind('inproc://delayed_msg')
        self.heartbeat_socket = self.context.socket(zmq.PAIR)
        self.heartbeat_socket.bind('inproc://heartbeat')

        self.workers = {}
        self.worker_online_locks = KeyedLocks()
        self.worker_msg_locks = KeyedLocks()

        self.processor_path = processor_path
        self.environ = environ

        self.worker_manager_thread = threading.Thread(
            target=self.thr_handle_worker_msg, name='workers', daemon=True)
        self.heartbeat_thread = threading.Thread(
            target=self.thr_heartbeat_timer, name='heartbeat', daemon=True)

    def thr_heartbeat_timer(self):
        logger.debug("Start heartbeat_timer")
        socket = self.context.socket(zmq.PAIR)
        socket.connect('inproc://heartbeat')
        while self.running:
            socket.send(b'')
            time.sleep(self.heartbeat)

    def check_online(self, name):
        with self.worker_online_locks[name]:
            if name in self.workers:
                return self.workers[name]

    def thr_handle_worker_msg(self):
        """
        Handle incoming commands from the Web backend.
        Check and start corresponding worker processes.
        This function runs in worker_manager_thread.
        """
        logger.debug("Start handle_worker_msg")
        socket = self.context.socket(zmq.PAIR)
        socket.connect('inproc://delayed_msg')
        while self.running:
            self.gc_process(socket)
            frames = socket.recv_multipart()
            header = frames[0]
            if header == b':manager-close':
                break
            elif header == b':heartbeat':
                continue
            logger.debug("handle_worker_msg recv: %s", frames)
            name = frames[0]
            worker = self.check_online(name)
            if not worker:
                if frames[2] == b':connect':
                    try:
                        worker = self.start_process(name)
                        socket.send_multipart([name, b':start'])
                    except TimeoutError as ex:
                        logger.warning('Start process timeout: %s', ex)
                        self.stop_process(name, True)
                else:
                    thr_backend_socket = self.context.socket(zmq.DEALER)
                    thr_backend_socket.setsockopt(zmq.IDENTITY, name)
                    thr_backend_socket.connect(self.backend_url)
                    thr_backend_socket.send_multipart([
                        b':cmd', frames[2],
                        serialize({'code': 502, 'error': 'worker not started'})
                    ])
                    thr_backend_socket.close()
                continue
            socket.send_multipart([name, b':cmd'] + list(frames[2:]))
        socket.close()

    def event_server_rpc(self):
        # copy=False will trigger Bad address (bundled\zeromq\src\fq.cpp:87)
        frames = self.server_socket.recv_multipart()
        logger.debug("event_server_rpc: %s", frames)
        self.delayed_msg_socket.send_multipart(frames)

    def event_delayed_msg(self):
        frames = self.delayed_msg_socket.recv_multipart()
        logger.debug("event_delayed_msg: %s", frames)
        # if frames[0][0] != ord(':'):
        if frames[0] in self.workers:
            self.backend_socket.send_multipart(frames)

    def event_heartbeat(self):
        self.heartbeat_socket.recv_multipart()
        self.delayed_msg_socket.send(b':heartbeat')

    def event_collector(self):
        #frames = self.backend_socket.recv_multipart(copy=False)
        frames = self.backend_socket.recv_multipart()
        self.logger.debug("event-collector: %s", frames)
        name = frames[0]
        msgtype = frames[1]
        if msgtype == b':cmd':
            self.server_socket.send_multipart([name, b''] + list(frames[2:]))
        elif msgtype in (b':connect', b':started'):
            with contextlib.suppress(KeyError):
                self.workers[name].started.set()
            if msgtype == b':started':
                self.server_socket.send_multipart((name, b'', b':started', b''))
        with contextlib.suppress(KeyError):
            self.workers[name].keep_alive()

    def gc_process(self, cmd_socket: 'zmq.Socket'):
        """
        Stop timeout and dead processes.
        This function runs in worker_manager_thread.
        """
        # logger.debug("gc_process")
        for name in tuple(self.workers.keys()):
            try:
                exit_status = self.workers[name].check(self.activity_timeout)
            except KeyError:
                continue
            if exit_status is None:
                continue
            cmd_socket.send_multipart((name, b':close'))
            self.stop_process(name, (exit_status == 'timeout'))

    def start_process(self, name):
        with self.worker_online_locks[name]:
            name_str = name.decode('utf-8', errors='replace')
            logger.info('starting process %s', name_str)
            if name in self.workers:
                return
            worker = WorkerProcess(
                self.context, name, self.backend_url, self.processor_path,
                self.environ
            )
            self.workers[name] = worker
            worker.start()
        return worker

    def stop_process(self, name, is_timeout=False):
        """
        Remove process.
        This function runs in worker_manager_thread.
        """
        with self.worker_online_locks[name]:
            logger.info('stopping (timeout=%s) %s', is_timeout, name)
            if name not in self.workers:
                return
            worker = self.workers[name]
            try:
                return_code = worker.close()
            except Exception:
                logger.exception('error closing worker %s', name)
            del self.workers[name]
            reason = None
            if is_timeout:
                reason = 'timeout'
            elif return_code == 0:
                reason = 'disconnect'
            elif return_code is not None:
                reason = 'error'
            with contextlib.suppress(zmq.ZMQError):
                self.server_socket.send_multipart((name, b'', b'', serialize({
                    'code': 502, 'error': 'worker closed'
                })))
            logger.info('stopped %s: %s', name, reason)

    def polling(self):
        self.poller.register(self.backend_socket, zmq.POLLIN)
        self.poller.register(self.heartbeat_socket, zmq.POLLIN)
        self.poller.register(self.server_socket, zmq.POLLIN)
        self.poller.register(self.delayed_msg_socket, zmq.POLLIN)
        logger.info('Session manager started polling')
        while self.running:
            for socket, event in self.poller.poll(timeout=1000):
                if socket == self.backend_socket:
                    self.event_collector()
                elif socket == self.heartbeat_socket:
                    self.event_heartbeat()
                elif socket == self.server_socket:
                    self.event_server_rpc()
                else:
                    self.event_delayed_msg()
        self.delayed_msg_socket.send(b':manager-close')

    def start(self):
        self.worker_manager_thread.start()
        self.heartbeat_thread.start()
        self.polling()

    def close(self):
        self.running = False

