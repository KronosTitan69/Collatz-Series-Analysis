#!/usr/bin/env python3
"""
Comprehensive Collatz Series Analysis Executor
This script systematically executes all Collatz analysis scripts and captures their outputs.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def execute_script(script_name, timeout=300):
    """Execute a script and capture its output"""
    print(f"Executing {script_name}...")
    
    try:
        # Set display for matplotlib
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
        
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd='/home/runner/work/Collatz-Series-Analysis/Collatz-Series-Analysis'
        )
        
        # Save output
        output_file = f"results/outputs/{script_name}_output.txt"
        with open(output_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")
        
        print(f"✓ {script_name} completed (exit code: {result.returncode})")
        return True, result.returncode, len(result.stdout), len(result.stderr)
        
    except subprocess.TimeoutExpired:
        print(f"⚠ {script_name} timed out after {timeout}s")
        return False, -1, 0, 0
    except Exception as e:
        print(f"✗ {script_name} failed: {e}")
        return False, -2, 0, 0

def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE COLLATZ SERIES ANALYSIS")
    print("="*80)
    
    # Ensure results directory exists
    Path("results/outputs").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # List of scripts to execute
    scripts = [
        'Code_Collatz_NN',
        'Collatz_UML_1',
        'Collatz_UML_2', 
        'Collatz_UML_3',
        'Collatz_UML_4',
        'Collatz_UML_6',
        'Collatz_UML_7',
        'Collatz_UML_8',
        'Collatz_UML_9',
        'Collatz_MC_1',
        'Collatz_MC_2',
        'Collatz_MC_3',
        'Collatz_MC_4',
        'Collatz_CV_1',
        'Collatz_CV_2',
        'Collatz_CV_4',
        'Colatz_CV_3'
    ]
    
    # Skip empty files
    scripts = [s for s in scripts if os.path.getsize(s) > 0]
    
    results = {}
    
    print(f"Found {len(scripts)} scripts to execute\n")
    
    for i, script in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] ", end="")
        success, exit_code, stdout_len, stderr_len = execute_script(script, timeout=180)
        
        results[script] = {
            'success': success,
            'exit_code': exit_code,
            'stdout_len': stdout_len,
            'stderr_len': stderr_len
        }
        
        time.sleep(1)  # Brief pause between executions
    
    # Generate summary report
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Successfully executed: {successful}/{total} scripts")
    print(f"Success rate: {successful/total*100:.1f}%\n")
    
    # Detailed results
    for script, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {script:<20} Exit: {result['exit_code']:>3} "
              f"Out: {result['stdout_len']:>6} chars Err: {result['stderr_len']:>6} chars")
    
    # Save summary
    with open("results/execution_summary.txt", 'w') as f:
        f.write("COLLATZ SERIES ANALYSIS - EXECUTION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total scripts: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
        f.write(f"Success rate: {successful/total*100:.1f}%\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*50 + "\n")
        for script, result in results.items():
            f.write(f"{script}: {'SUCCESS' if result['success'] else 'FAILED'} "
                   f"(exit: {result['exit_code']}, stdout: {result['stdout_len']}, "
                   f"stderr: {result['stderr_len']})\n")
    
    print(f"\nExecution summary saved to results/execution_summary.txt")
    print("Individual outputs saved to results/outputs/")

if __name__ == "__main__":
    main()