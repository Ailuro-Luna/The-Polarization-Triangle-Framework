import argparse
import subprocess
import sys
import os
from pathlib import Path

# --- Configuration ---
SHORT_MODE_STEPS = 10
LONG_MODE_STEPS = 200

# Steps specifically for verification tests (can be shorter)
VERIFICATION_SHORT_STEPS = 5
VERIFICATION_LONG_STEPS = 20

# Parameters for specific tests (can be reduced in short mode)
SHORT_MORALITY_RATES = "0.1 0.9"
LONG_MORALITY_RATES = "0.1 0.3 0.5 0.7 0.9"

SHORT_ALPHABETA_BETA_STEPS = 3
LONG_ALPHABETA_BETA_STEPS = 10
SHORT_ALPHABETA_NUM_RUNS = 2
LONG_ALPHABETA_NUM_RUNS = 10
# --- End Configuration ---

def run_command(command, description):
    """Executes a command, prints status, and returns success status."""
    print(f"--- Running: {description} ---")
    print(f"Executing: {' '.join(command)}")
    try:
        # Ensure the command is run from the project root directory
        project_root = Path(__file__).parent.parent.parent # Adjust if script location changes
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=project_root, shell=(os.name == 'nt')) # Use shell=True on Windows for python -m
        print(f"--- Success: {description} ---")
        print("Output (last 500 chars):", result.stdout[-500:])
        if result.stderr:
            print("Stderr (last 500 chars):", result.stderr[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- FAILED: {description} ---", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Stderr:{e.stderr}", file=sys.stderr)
        print(f"Stdout:{e.stdout}", file=sys.stderr)
        # Decide whether to exit or continue
        # sys.exit(f"Execution failed for: {description}")
        print(f"Continuing with next tests despite failure in {description}...", file=sys.stderr)
        return False
    except Exception as e:
        print(f"--- UNEXPECTED ERROR during: {description} ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"Continuing with next tests despite error in {description}...", file=sys.stderr)
        return False
    finally:
        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description="Run all tests for the Polarization Triangle Framework.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["short", "long"],
        required=True,
        help="Execution mode: 'short' for quick checks, 'long' for full simulations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/all_tests",
        help="Base directory to store results for all tests."
    )
    args = parser.parse_args()

    steps = SHORT_MODE_STEPS if args.mode == "short" else LONG_MODE_STEPS
    verification_steps = VERIFICATION_SHORT_STEPS if args.mode == "short" else VERIFICATION_LONG_STEPS
    morality_rates = SHORT_MORALITY_RATES if args.mode == "short" else LONG_MORALITY_RATES
    alphabeta_beta_steps = SHORT_ALPHABETA_BETA_STEPS if args.mode == "short" else LONG_ALPHABETA_BETA_STEPS
    alphabeta_num_runs = SHORT_ALPHABETA_NUM_RUNS if args.mode == "short" else LONG_ALPHABETA_NUM_RUNS

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = [] # List to store (description, success_status)

    # Define commands using python -m to ensure correct module resolution
    executable = sys.executable # Use the same python executable that runs this script

    # 1. Basic Simulation
    basic_output = base_output_dir / "batch_results"
    cmd_basic = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "basic",
        "--output-dir", str(basic_output),
        "--steps", str(steps)
    ]
    success = run_command(cmd_basic, "Basic Simulation")
    results_summary.append(("Basic Simulation", success))

    # 2. Morality Test
    morality_output = base_output_dir / "morality"
    cmd_morality = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "morality",
        "--output-dir", str(morality_output),
        "--steps", str(steps),
        "--morality-rates"
    ] + morality_rates.split()
    success = run_command(cmd_morality, "Morality Rate Test")
    results_summary.append(("Morality Rate Test", success))

    # 3. Model Parameters Test
    params_output = base_output_dir / "params"
    cmd_params = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "model-params",
        "--output-dir", str(params_output),
        "--steps", str(steps)
    ]
    success = run_command(cmd_params, "Model Parameters Test")
    results_summary.append(("Model Parameters Test", success))

    # 4. Activation Analysis
    activation_output = base_output_dir / "activation"
    cmd_activation = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "activation",
        "--output-dir", str(activation_output),
        "--steps", str(steps)
    ]
    success = run_command(cmd_activation, "Activation Analysis")
    results_summary.append(("Activation Analysis", success))

    # 5. Verification - Alpha
    verification_alpha_output = base_output_dir / "verification" / "alpha_verification"
    cmd_verify_alpha = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "verification", "--verification-type", "alpha",
        "--output-dir", str(verification_alpha_output.parent), # main expects parent dir
        "--alpha-min", "-0.5", "--alpha-max", "1.5"
    ]
    success = run_command(cmd_verify_alpha, "Verification - Alpha")
    results_summary.append(("Verification - Alpha", success))

    # 6. Verification - AlphaBeta
    verification_alphabeta_output = base_output_dir / "verification" / "alphabeta_verification"
    cmd_verify_alphabeta = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "verification", "--verification-type", "alphabeta",
        "--output-dir", str(verification_alphabeta_output.parent), # main expects parent dir
        "--low-alpha", "0.5", "--high-alpha", "1.0", # Reduced range for testing
        "--beta-min", "0.5", "--beta-max", "1.5",   # Reduced range for testing
        "--beta-steps", str(alphabeta_beta_steps),
        "--morality-rate", "0.3",
        "--num-runs", str(alphabeta_num_runs),
        "--steps", str(verification_steps) # Simulation steps within each run
    ]
    success = run_command(cmd_verify_alphabeta, "Verification - AlphaBeta")
    results_summary.append(("Verification - AlphaBeta", success))

    # 7. Verification - Agent Interaction
    verification_agent_output = base_output_dir / "verification" / "agent_interaction"
    cmd_verify_agent = [
        executable, "-m", "polarization_triangle.main",
        "--test-type", "verification", "--verification-type", "agent_interaction",
        "--output-dir", str(verification_agent_output.parent), # main expects parent dir
        "--steps", str(verification_steps)
    ]
    success = run_command(cmd_verify_agent, "Verification - Agent Interaction")
    results_summary.append(("Verification - Agent Interaction", success))

    # 8. Visualization (Optional - Example: AlphaBeta)
    # Ensure the results directory exists before trying to visualize
    alphabeta_results_file = verification_alphabeta_output / "alphabeta_results.csv"
    alphabeta_viz_output_dir = verification_alphabeta_output / "visualizations"

    if alphabeta_results_file.exists():
        cmd_visualize_alphabeta = [
            executable, "-m", "polarization_triangle.scripts.visualize_alphabeta",
            "--result-file", str(alphabeta_results_file),
            "--output-dir", str(alphabeta_viz_output_dir),
            "--num-runs", str(alphabeta_num_runs)
        ]
        success = run_command(cmd_visualize_alphabeta, "Visualize AlphaBeta Results")
        results_summary.append(("Visualize AlphaBeta Results", success))
    else:
        print(f"--- Skipping: Visualize AlphaBeta Results (Results file not found: {alphabeta_results_file}) ---")
        results_summary.append(("Visualize AlphaBeta Results", "Skipped")) # Mark as skipped


    print("\n" + "=" * 15 + " Test Summary " + "=" * 15)
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for description, status in results_summary:
        if status == True:
            status_text = "PASSED"
            passed_count += 1
        elif status == False:
            status_text = "FAILED"
            failed_count += 1
        else:
            status_text = "SKIPPED"
            skipped_count += 1
        print(f"{description:<35}: {status_text}")

    print("-" * 44)
    print(f"Total Tests: {len(results_summary)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 44)

    if failed_count > 0:
        print("\nSome tests failed.")
        # sys.exit(1) # Optionally exit with error code if any test fails

if __name__ == "__main__":
    main() 