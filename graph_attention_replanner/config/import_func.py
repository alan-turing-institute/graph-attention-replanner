from rl4co.envs.routing import (
    MTSPEnv_Type1,
    MTSPEnv_Type2,
    MTSPEnv_Type3,
    MTSPEnv_Type4,
    MTSPEnv_Type5,
)

from rl4co.envs.routing import (
    MTSPGenerator_Type1,
    MTSPGenerator_Type2,
    MTSPGenerator_Type3,
    MTSPGenerator_Type4,
    MTSPGenerator_Type5,
)


def get_env(problem_type):
    envs = {
        1: MTSPEnv_Type1,
        2: MTSPEnv_Type2,
        3: MTSPEnv_Type3,
        4: MTSPEnv_Type4,
        5: MTSPEnv_Type5,
    }
    return envs[problem_type]


def get_generator(problem_type):
    gens = {
        1: MTSPGenerator_Type1,
        2: MTSPGenerator_Type2,
        3: MTSPGenerator_Type3,
        4: MTSPGenerator_Type4,
        5: MTSPGenerator_Type5,
    }
    return gens[problem_type]

