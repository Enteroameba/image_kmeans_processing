import subprocess

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

for package in requirements:
    try:
        subprocess.check_call(["pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Skipping installation of {package} due to an error.")
print("packages installation finished")