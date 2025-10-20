from openpi.training import config as openpi_config
from openpi.policies import policy_config
import numpy as np
from openpi_client import image_tools
import math

# def _quat2axisangle(quat):
#     """
#     Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
#     """
#     # clip quaternion
#     if quat[3] > 1.0:
#         quat[3] = 1.0
#     elif quat[3] < -1.0:
#         quat[3] = -1.0

#     den = np.sqrt(1.0 - quat[3] * quat[3])
#     if math.isclose(den, 0.0):
#         # This is (close to) a zero degree rotation, immediately return
#         return np.zeros(3)

#     return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# def obs_to_pi_zero_input(obs):
#     img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
#     wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
#     img = image_tools.convert_to_uint8(
#         image_tools.resize_with_pad(img, 224, 224)
#     )
#     wrist_img = image_tools.convert_to_uint8(
#         image_tools.resize_with_pad(wrist_img, 224, 224)
#     )
    
#     obs_pi_zero = {
#                     "observation/image": img,
#                     "observation/wrist_image": wrist_img,
#                     "observation/state": np.concatenate(
#                         (
#                             obs["robot0_eef_pos"],
#                             _quat2axisangle(obs["robot0_eef_quat"]),
#                             obs["robot0_gripper_qpos"],
#                         )
#                     ),
#                     "prompt": "do something",
#                 }
#     return obs_pi_zero

# agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
# num_samples = 10
# for _ in range(num_samples):
#     print(f" Sampling pi0_libero_mine policy")
#     action = agent_dp.infer(dummy_obs_pi_zero)["actions"]
#     print(f" Action: {action}")

dummy_obs = {
    "agentview_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
    "robot0_eye_in_hand_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
    "robot0_eef_pos": np.random.rand(3),
    "robot0_eef_quat": np.random.rand(4),
    "robot0_gripper_qpos": np.random.rand(2),
}

# dummy_obs_pi_zero = obs_to_pi_zero_input(dummy_obs)
# print(dummy_obs_pi_zero)

from networks.nets import Pi0Actor
config = openpi_config.get_config("pi0_libero_mine")
checkpoint_dir = '/raid/users/yajatyadav/checkpoints/pi0_all_libero_but_10_flipped_train_split/pi0_all_libero_but_10_flipped_train_split__batch_64_steps_30k/10000'
pi0actor = Pi0Actor(checkpoint_dir=checkpoint_dir, config_name='pi0_libero_mine', action_horizon=1)
action = pi0actor(dummy_obs, "do something")
import pdb; pdb.set_trace()