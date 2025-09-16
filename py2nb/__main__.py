#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа для запуска конвертера как модуля
"""

import sys
from pathlib import Path

# Импортируем функции из __init__.py
from . import main

if __name__ == "__main__":
    main()
