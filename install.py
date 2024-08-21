import os
import yaml

with open("environment.yaml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

os.system("source /home/matthew.a.chan/latent-diffusion/env/bin/activate")
for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
      for lib in dependency['pip']:
        os.system(f"pip install {lib}")