import subprocess
import sys
import os


def run_script(script_name):
    print(f"\n{'=' * 50}")
    print(f"Running {script_name}...")
    print(f"{'=' * 50}\n")

    try:
        # Run the script and capture its output
        result = subprocess.run([sys.executable, script_name],
                                capture_output=True,
                                text=True)

        # Print the output
        print(result.stdout)

        # Check if there were any errors
        if result.stderr:
            print("Errors:", result.stderr)

        if result.returncode != 0:
            print(f"Error: {script_name} failed with return code {result.returncode}")
            return False

        return True

    except Exception as e:
        print(f"Error running {script_name}: {str(e)}")
        return False


def main():
    # First run Crop_images2.py
    if not run_script("Crop_images2.py"):
        print("Failed to run Crop_images2.py. Stopping execution.")
        return

    # Then run ModelV2.py
    if not run_script("ModelV2.py"):
        print("Failed to run ModelV2.py. Stopping execution.")
        return

    # Finally run
    if not run_script("AI_API_Suggestions.py"):
        print("Failed to run AI_API_Suggestions.py. Stopping execution.")
        return

    if not run_script("Final_File_Creation.py"):
        print("Failed to run Final_File_Creation.py. Stopping execution.")
        return


    print("\nAll scripts completed successfully!")


if __name__ == "__main__":
    main()