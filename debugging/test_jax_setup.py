import jax
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# If you see GPU but want to force CPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''