#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polarization Triangle Framework Main Entry File
Provides command line interface to run various simulation tests
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polarization_triangle.experiments.batch_runner import batch_test
from polarization_triangle.experiments.morality_test import batch_test_morality_rates
from polarization_triangle.experiments.model_params_test import batch_test_model_params
from polarization_triangle.experiments.activation_analysis import analyze_activation_components


def main():
    """
    Main entry function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Polarization Triangle Framework Simulation")
    parser.add_argument("--test-type", 
                        choices=["basic", "morality", "model-params", "activation", "verification"],
                        default="basic",
                        help="Type of test to run")
    parser.add_argument("--output-dir", type=str, default="batch_results",
                       help="Output directory name")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of simulation steps")
    parser.add_argument("--morality-rates", type=float, nargs="+", 
                       default=[0.2, 0.4, 0.6, 0.8],
                       help="List of morality rates for morality test")
    parser.add_argument("--verification-type", 
                        choices=["alpha", "alphabeta", "agent_interaction", "all"],
                        default="alpha",
                        help="Verification type (used only when test-type is verification)")
    parser.add_argument("--alpha-min", type=float, default=-1.0,
                       help="Minimum alpha value for analysis (used only when verification-type is alpha)")
    parser.add_argument("--alpha-max", type=float, default=2.0,
                       help="Maximum alpha value for analysis (used only when verification-type is alpha)")
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                       help="Alpha values for alphabeta analysis")
    parser.add_argument("--beta-min", type=float, default=0.1,
                       help="Minimum beta value for alphabeta analysis")
    parser.add_argument("--beta-max", type=float, default=2.0,
                       help="Maximum beta value for alphabeta analysis")
    parser.add_argument("--beta-steps", type=int, default=10,
                       help="Number of beta steps for alphabeta analysis")
    parser.add_argument("--morality-rate", type=float, default=0.0,
                       help="Morality rate for alphabeta analysis (0.0-1.0)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of simulation runs per parameter combination for alphabeta analysis")
    
    args = parser.parse_args()
    
    # Run different tests based on test type
    if args.test_type == "basic":
        print("Running basic simulation...")
        # Use batch_test from experiments module
        batch_test(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "morality":
        print(f"Running morality rate test, morality rates: {args.morality_rates}...")
        # Use batch_test_morality_rates from experiments module
        batch_test_morality_rates(output_dir=args.output_dir, steps=args.steps,
                         morality_rates=args.morality_rates)
        
    elif args.test_type == "model-params":
        print("Running model parameters test...")
        # Use batch_test_model_params from experiments module
        batch_test_model_params(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "activation":
        print("Running activation component analysis...")
        # run analyze_activation_components from experiments module
        analyze_activation_components(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "verification":
        print(f"Running verification analysis, type: {args.verification_type}...")
        
        # Function to run alpha verification
        def run_alpha_verification():
            from polarization_triangle.verification.alpha_analysis import AlphaVerification
            verification = AlphaVerification(
                alpha_range=(args.alpha_min, args.alpha_max),
                output_dir=os.path.join(args.output_dir, "alpha_verification")
            )
            verification.run()
            
        # Function to run alphabeta verification    
        def run_alphabeta_verification():
            from polarization_triangle.scripts.run_alphabeta_verification import run_alphabeta_verification
            output_dir = os.path.join(args.output_dir, "alphabeta_verification")
            run_alphabeta_verification(
                output_dir=output_dir,
                steps=args.steps,
                # low_alpha=args.low_alpha,
                # high_alpha=args.high_alpha,
                beta_min=args.beta_min,
                beta_max=args.beta_max,
                beta_steps=args.beta_steps,
                morality_rate=args.morality_rate,
                num_runs=args.num_runs
            )
            
        # Function to run agent interaction verification
        def run_agent_interaction_verification():
            from polarization_triangle.verification.agent_interaction_verification import main as agent_verification_main
            output_dir = os.path.join(args.output_dir, "agent_interaction_verification")
            # os.makedirs(output_dir, exist_ok=True)
            # agent_verification_main(output_dir=output_dir, num_steps=args.steps)
            agent_verification_main(output_dir=output_dir, num_steps=1)
            
        # Run verification based on type
        if args.verification_type == "alpha":
            run_alpha_verification()
        elif args.verification_type == "alphabeta":
            run_alphabeta_verification()
        elif args.verification_type == "agent_interaction":
            run_agent_interaction_verification()
        elif args.verification_type == "all":
            print("Running all verification types...")
            run_alpha_verification()
            run_alphabeta_verification()
            run_agent_interaction_verification()
    
    print("Completed!")


if __name__ == "__main__":
    main()
