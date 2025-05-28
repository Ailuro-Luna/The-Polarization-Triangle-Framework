#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polarization Triangle Framework Main Entry

Simple entry point for running experiments.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polarization_triangle.scripts.run_experiment import main

if __name__ == "__main__":
    main()
