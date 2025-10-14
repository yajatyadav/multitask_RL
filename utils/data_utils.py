import numpy as np

# used by the dataloader during training / in the eval script before feeding into the policy

# utility func b/c openvla dataloader wants one dataset in a mixture to be the "primary" one and have weight 1.0
def ratios_to_odds_mixture(ratios):
    keys = sorted(ratios.keys())
    first_val = ratios[keys[0]]
    new_dict = {keys[0]: 1.0}
    for key in keys[1:]:
        new_dict[key] = ratios[key] / first_val
    return new_dict


# language encoder that encodes str into vector; used by the dataloader during training / in the eval script before feeding into the policy
import tensorflow_hub as hub
import tensorflow_text
MUSE_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
class MuseEmbedding:
    @staticmethod
    def encode(strings):
        return MUSE_MODEL(strings).numpy()


## normalization stats
# these are norm stats from RETAIN-LIbero's pretraining data (which is libero-{90, goal, object, spatial}, with a few tasks held out from each)
# while not the exact same as the multitask_RL project, these stats should be close enough to be used for normalization
LIBERO_NORM_STATS_KEY = "norm_stats__all_libero_but_10_train_split"


NORM_STATS = {
"norm_stats__all_libero_but_10_train_split": {
"state": {
    "mean": [
    -0.07594718039035797,
    0.01788640394806862,
    0.9031365513801575,
    2.9963114261627197,
    -0.11735942214727402,
    -0.09229698777198792,
    0.02637086622416973,
    -0.026728888973593712,
    ],
    "std": [
    0.11286084353923798,
    0.14507046341896057,
    0.2954331636428833,
    0.3924177587032318,
    0.9082265496253967,
    0.31110888719558716,
    0.014611119404435158,
    0.014481362886726856,
    ],
    "q01": [
    -0.40484070825874807,
    -0.2767059429228306,
    0.1121419694436714,
    1.4359511102080345,
    -2.720485407066345,
    -1.2217254063606264,
    0.0016512279755901546,
    -0.040038076904509216,
    ],
    "q99": [
    0.1257090463906526,
    0.32582998911738403,
    1.2557504985231904,
    3.697139627039432,
    2.3124017268180843,
    0.5870746158599851,
    0.04028308051750064,
    -0.001706432364764622,
    ]
},
"actions": {
    "mean": [
    0.0896475613117218,
    0.0353005975484848,
    -0.9628857970237732,
    -2.993234872817993,
    0.12035975605249405,
    0.08618046343326569,
    0.020620649680495262,
    ],
    "std": [
    0.30294424295425415,
    0.3951551020145416,
    0.5384603142738342,
    0.3950735330581665,
    0.9109124541282654,
    0.3203604817390442,
    0.9997873902320862,
    ],
    "q01": [
    -0.6131696268081666,
    -0.9530494272947312,
    -2.0260506366729736,
    -3.671418887233734,
    -2.3304990768432616,
    -0.6162669191360475,
    -1.0,
    ],
    "q99": [
    0.9955515738487244,
    1.0174857356309892,
    0.4062486431121828,
    -1.433714524137974,
    2.7279676017761236,
    1.2093748165130611,
    0.9996,
    ]
}
}
}

# proprio normalization: use mean/std for first 6 dims, sgn(.) for last 2 gripper dims
def normalize_proprio(proprio):
    assert proprio.dtype == np.float32
    proprio = np.array(proprio)
    proprio_norm_stats = NORM_STATS[LIBERO_NORM_STATS_KEY]["state"]
    mean, std = proprio_norm_stats["mean"], proprio_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)

    ## apply to all dims
    proprio = (proprio - mean) / (std + 1e-8)
    return proprio


# binarization is handled by the dataloader OR the eval script
def normalize_action(action):
    assert action.dtype == np.float32
    # same idea, except only normalize first 6 dims
    action = np.array(action)
    action_norm_stats = NORM_STATS[LIBERO_NORM_STATS_KEY]["actions"]
    mean, std = action_norm_stats["mean"], action_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)

    ## only normalize first 6 dims: do nothing w/ last 2 as they already should've been binarized by dataloader
    action[:6] = (action[:6] - mean[:6]) / (std[:6] + 1e-8)
    return action
    

def unnormalize_action(action):
    assert action.dtype == np.float32
    action_norm_stats = NORM_STATS[LIBERO_NORM_STATS_KEY]["actions"]
    mean, std = action_norm_stats["mean"], action_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)
    
    ## only unnormalize first 6 dims: do nothing w/ last 2 as it will be binarized by eval script
    action[:6] = (action[:6] * (std[:6] + 1e-8)) + mean[:6]
    return action

# puts images in range [-1, 1]
def normalize_image(image):
    assert image.dtype == np.uint8
    image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
    return image


def unnormalize_image(image):
    assert image.dtype == np.float32
    image = (image + 1.0) / 2.0 * 255.0
    image = image.astype(np.uint8)
    return image