Implementation of all methods and experiments in multitask_RL.


Setup:
1. Get LIBERO submodule: git submodule update --init --recursive
2. Setup all packages: uv sync
3. To run libero experiments, you need to:
    - install libero in editable mode: pip install -e libero (actually this should be taken care of by uv sync)
    - set the MULTITASK_RL_REPO_ROOT_DIR env variable so the test scripts can properly append to the pythonpath..

Note:
- at the VERY TOP of each file you will run, place: import tensorflow as tf
tf.config.set_visible_devices([], "GPU"), so that the dataloader doesn't take GPU mem


When the LIBERO submodule github gets updated, run git submodule update --remote --merge, in order to pull in new changes!

Stuff that is KILLING the dataloading speed:
1. image augmentation stack
2. my custom normalization functions (need to refactor these into the TFDS pipeline as transforms)