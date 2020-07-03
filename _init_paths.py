import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
print ('this dir: {}'.format(this_dir))

# Add lib to PYTHONPATH src
lib_path = osp.join(this_dir, 'src')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))

# Add lib to PYTHONPATH src/image_reader
lib_path = osp.join(this_dir, 'src', 'image_reader')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))

# Add lib to PYTHONPATH src/image_reader
lib_path = osp.join(this_dir, 'src', 'datasets')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))

# Add lib to PYTHONPATH model
lib_path = osp.join(this_dir, 'models')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))

# Add lib to PYTHONPATH model/operations
lib_path = osp.join(this_dir, 'models','operations')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))

# Add lib to PYTHONPATH tools
lib_path = osp.join(this_dir, 'tools')
add_path(lib_path)
print ('add {} to PYTHONPATH'.format(lib_path))



