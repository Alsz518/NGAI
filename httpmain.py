#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import base64
import secrets
import hashlib
import logging
import argparse
import collections

import pytz
import tomli
import bottle
import nacl.utils
import nacl.pwhash
import nacl.secret
from nacl.exceptions import CryptoError

from btrafficai.client import BaseChatBotClient, RPCError


app = application = bottle.default_app()
script_path = os.path.abspath(os.path.dirname(__file__))

CONFIG = {}
TIMEZONE = pytz.timezone('Asia/Shanghai')
USERS = {
    'admin': 'zone@1234',
}

COOKIE_TTL = 3600 * 2
CAPTCHA_LENGTH = 4
CAPTCHA_ALPHABET = 'abcdefhikmnprtuvwxyABCDEFGHJKLMNPQRTUVWXY123456789'
CAPTCHA_TTL = 300


def make_insert(d):
    keys, values = zip(*d.items())
    return ', '.join(keys), ', '.join('?' * len(values)), values

def make_update(d):
    keys, values = zip(*d.items())
    return ', '.join(k + '=?' for k in keys), values


def load_config(filename):
    with open(filename, 'rb') as f:
        config_file = tomli.load(f)
    CONFIG.update(config_file['http'])
    CONFIG['timezone'] = pytz.timezone(CONFIG['timezone'])


@app.error(400)
@app.error(403)
@app.error(404)
@app.error(406)
@app.error(500)
@app.error(502)
@app.error(503)
def error_handler(error):
    bottle.response.content_type = 'application/json'
    return json.dumps({"code": error.status_code, "error": error.body})


class BadToken(Exception):
    def __init__(self, token):
        self.token = token


class ExpiredToken(BadToken):
    def __init__(self, token, content):
        self.token = token
        self.content = content


class InvalidSaltToken(BadToken):
    def __init__(self, token, content):
        self.token = token
        self.content = content


TimestampedValue = collections.namedtuple(
    'TimestampedValue', ('timestamp', 'salt', 'value'))


class SecretSigner:
    def __init__(self, secret_key: bytes):
        self.secret_box = nacl.secret.SecretBox(secret_key)

    @classmethod
    def salt_hash(cls, text: str) -> bytes:
        return hashlib.blake2b(text.encode('utf-8'), digest_size=4).digest()

    def encrypt(self, text: str, salt=None) -> str:
        time_ms = int(time.time() * 1000)
        salt_value = self.salt_hash(salt) if salt is not None else b''
        payload = time_ms.to_bytes(8, 'little') + salt_value + text.encode('utf-8')
        encrypted = self.secret_box.encrypt(payload)
        return base64.urlsafe_b64encode(encrypted).decode('ascii')

    def decrypt_raw(self, cipher: str, has_salt=False) -> TimestampedValue:
        try:
            encrypted = base64.urlsafe_b64decode(cipher)
            payload = self.secret_box.decrypt(encrypted)
        except (ValueError, CryptoError):
            raise BadToken(cipher)
        timestamp = int.from_bytes(payload[:8], 'little') / 1000
        if has_salt:
            salt = payload[8:12]
            value = payload[12:].decode('utf-8')
        else:
            salt = None
            value = payload[8:].decode('utf-8')
        return TimestampedValue(timestamp, salt, value)

    def decrypt(self, cipher: str, ttl=None, salt=None) -> str:
        """
        Decrypt the token and check ttl. Unit is second.
        If salt is None, there is no salt in the cipher;
        If salt is True, don't check the salt.
        """
        result = self.decrypt_raw(cipher, (salt is not None))
        if ttl is not None and result.timestamp + ttl < time.time():
            raise ExpiredToken(cipher, result)
        if (salt is not None and salt is not True
            and self.salt_hash(salt) != result.salt):
            raise InvalidSaltToken(cipher, result)
        return result.value

    def verify(self, cipher: str, plain: str, ttl=None, salt=None) -> bool:
        try:
            value = self.decrypt(cipher, ttl, salt)
        except BadToken:
            return False
        return secrets.compare_digest(plain, value)

    def token(self, salt=None) -> str:
        return self.encrypt('', salt)

    def verify_token(self, token: str, ttl=None, salt='') -> bool:
        return self.verify(token, '', ttl, salt)


@app.post('/login')
def login():
    try:
        args = json.load(bottle.request.body)
    except json.JSONDecodeError:
        raise bottle.HTTPError(406, "invalid JSON")
    try:
        if not secrets.compare_digest(args['password'], USERS[args['username']]):
            raise bottle.HTTPError(403, "invalid user/password")
    except KeyError:
        raise bottle.HTTPError(403, "invalid user/password")
    signer = SecretSigner(CONFIG['secret_key'])
    bottle.response.set_cookie(
        "user_token", signer.encrypt(args['username']),
        expires=(int(time.time()) + COOKIE_TTL),
        httponly=True, same_site='strict'
    )
    return {'username': args['username']}


def check_login(anonymous=False):
    token = bottle.request.get_cookie('user_token')
    signer = SecretSigner(CONFIG['secret_key'])
    if not token:
        if anonymous:
            return None
        raise bottle.HTTPError(403, 'not logged in')
    try:
        user_id = signer.decrypt(token, COOKIE_TTL)
    except BadToken:
        if anonymous:
            return None
        raise bottle.HTTPError(403, 'not logged in')
    if user_id not in USERS:
        if anonymous:
            return None
        raise bottle.HTTPError(403, 'not logged in')
    return user_id


def get_request_param(key: str):
    # return bottle.request.params.get(key, '').encode('latin1').decode('utf-8')
    return bottle.request.params.get(key, '')


def get_request_boolean_param(key: str):
    param = get_request_param(key)
    return (param and param != '0')


def get_chatbot_client(client_id=None):
    return BaseChatBotClient(CONFIG['manager_url'], client_id)



@app.post('/chatbot/start')
def chatbot_start():
    # user_id = check_login(True)
    with get_chatbot_client() as client:
        client_id = client.client_id
        try:
            client.start()
        except RPCError as ex:
            raise bottle.HTTPError(ex.code, ex.error)
    return {'code': 200, 'client_id': client_id}


def _common_chatbot_cmd(method, **kwargs):
    client_id = get_request_param("client_id")
    if not client_id:
        raise bottle.HTTPError(400, "client_id not supplied")
    with get_chatbot_client(client_id) as client:
        try:
            response = client.zmq_rpc(method, **kwargs)
        except RPCError as ex:
            raise bottle.HTTPError(ex.code, ex.error)
    return {'code': 200, 'client_id': client_id, 'data': response}


@app.post('/chatbot/respond')
def chatbot_respond():
    # user_id = check_login(True)
    return _common_chatbot_cmd(
        "respond", message=get_request_param("message"))


@app.post('/chatbot/stop')
def chatbot_respond():
    # user_id = check_login(True)
    return _common_chatbot_cmd("stop")


@app.post('/chatbot/reset')
def chatbot_respond():
    # user_id = check_login(True)
    return _common_chatbot_cmd("reset")


@app.post('/chatbot/get_history')
def chatbot_get_history():
    # user_id = check_login(True)
    return _common_chatbot_cmd(
        "get_history", full=get_request_boolean_param("full"))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname).1s:%(name)s] %(message)s'
    )

    parser = argparse.ArgumentParser(description='Chat bot HTTP server')
    parser.add_argument(
        "-l", "--host",
        default="0.0.0.0", help="listen address")
    parser.add_argument(
        "-p", "--port",
        default=8082, type=int, help="listen port")
    parser.add_argument(
        "-c", "--config",
        default="config.toml", help="config file")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", help="show debug log")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('btrafficai.client').setLevel(logging.DEBUG)

    load_config(args.config)
    app.run(host=args.host, port=args.port, server='auto')
