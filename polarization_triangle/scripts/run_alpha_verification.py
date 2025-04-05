#!/usr/bin/env python
"""
Script to run alpha verification analysis.

This script analyzes the behavior of the simplified equation dz/dt = -z + tanh(alpha*z) when beta=0.
It specifically examines the dynamics of dz/dt and z as alpha varies between -1 and 2.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from polarization_triangle.verification.alpha_analysis import AlphaVerification


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run alpha verification analysis for the Polarization Triangle Framework"
    )
    
    parser.add_argument(
        "--alpha-min", 
        type=float, 
        default=-1.0,
        help="Minimum value of alpha (default: -1.0)"
    )
    
    parser.add_argument(
        "--alpha-max", 
        type=float, 
        default=2.0,
        help="Maximum value of alpha (default: 2.0)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (default: project_root/results/verification)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create verification instance
    verification = AlphaVerification(
        alpha_range=(args.alpha_min, args.alpha_max),
        output_dir=args.output_dir
    )
    
    # Run verification analysis
    verification.run()
    
    print("Alpha verification analysis completed!")


if __name__ == "__main__":
    main() 