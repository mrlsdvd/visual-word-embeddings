import os, sys

# Relevant paths
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
models_path = os.path.join(project_path, 'models')
core_path = os.path.join(project_path, 'core')
core_models_path = os.path.join(core_path, 'models')
backbones_path = os.path.join(project_path, 'backbones')
data_path = os.path.join(project_path, 'data')
processed_data_path = os.path.join(data_path, 'processed')
utils_path = os.path.join(core_path, 'util')
