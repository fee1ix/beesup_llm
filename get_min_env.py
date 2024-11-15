import os
import subprocess
import yaml

def generate_requirements(repo_path):
    subprocess.run(["pipreqs", repo_path, "--force"], check=True)

def get_conda_env():
    result = subprocess.run(["conda", "env", "export", "--no-builds"], capture_output=True, text=True, check=True)
    return yaml.safe_load(result.stdout)

def filter_conda_packages(conda_env, pip_packages):
    conda_deps = conda_env.get("dependencies", [])
    pip_deps = {pkg.split("==")[0] for pkg in pip_packages}
    filtered_deps = [pkg for pkg in conda_deps if isinstance(pkg, str) and pkg.split("=")[0] not in pip_deps]
    return filtered_deps

def create_conda_yaml(filtered_deps, pip_packages, output_file):
    env = {
        "name": "minimal-environment", 
        "channels": ["conda-forge", "defaults"],
        "dependencies": filtered_deps + [{"pip": pip_packages}]
    }
    with open(output_file, "w") as f:
        yaml.dump(env, f)

if __name__ == "__main__":
    repo_path = ""
    output_file = "minimal_environment.yaml"
    
    generate_requirements(repo_path)
    with open(os.path.join(repo_path, "requirements.txt")) as f:
        pip_packages = [line.strip() for line in f.readlines()]
    
    conda_env = get_conda_env()
    filtered_deps = filter_conda_packages(conda_env, pip_packages)
    
    create_conda_yaml(filtered_deps, pip_packages, output_file)
    print(f"Environment YAML written to {output_file}")