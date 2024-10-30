#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging

from btrafficai import worker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname).1s:%(name)s] %(message)s'
)

if __name__ == '__main__':
    w = worker.Worker(*sys.argv[1:])
    w.start()
