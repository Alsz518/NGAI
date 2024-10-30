#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import atexit
import argparse
import subprocess

processes = []

def cleanup():
    for p in processes:
        p.terminate()
        p.kill()


def main():
    parser = argparse.ArgumentParser(description='Run all the processes')
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

    cmd_manager = [
        sys.executable, "session_manager.py",
        "-c", args.config
    ]
    if args.verbose:
        cmd_manager.append("-v")
    processes.append(subprocess.Popen(cmd_manager))

    cmd_http = [
        sys.executable, "httpmain.py",
        "-l", args.host,
        "-p", str(args.port),
        "-c", args.config,
    ]
    if args.verbose:
        cmd_http.append("-v")
    proc_http = subprocess.Popen(cmd_http)
    processes.append(proc_http)

    atexit.register(cleanup)
    try:
        proc_http.wait()
    except KeyboardInterrupt:
        proc_http.terminate()


if __name__ == '__main__':
    main()
