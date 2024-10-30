#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname).1s:%(name)s:%(threadName)s] %(message)s'
)


def load_config(config_file):
    import tomli
    with open(config_file, 'rb') as f:
        config = tomli.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='User session manager')
    parser.add_argument(
        "-s", "--server-url", help="command server url")
    parser.add_argument(
        "-c", "--config", default="config.toml", help="read config")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show debug log")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger('session_manager').setLevel(logging.DEBUG)
    config = {}
    if args.config:
        config = load_config(args.config)
    server_url = args.server_url
    if not server_url:
        server_url = config.get("http", {}).get(
            "manager_url", "tcp://127.0.0.1:7890")

    # import main module after setting environment variables
    from btrafficai.manager import SessionManager

    session_manager = SessionManager(
        server_url,
        os.path.join(os.path.dirname(__file__), 'session_worker.py'),
        environ=config.get('environ', {})
    )
    try:
        session_manager.start()
    except KeyboardInterrupt:
        session_manager.close()


if __name__ == '__main__':
    main()
