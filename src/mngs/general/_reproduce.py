#!/usr/bin/env python3

import mngs


################################################################################
## Reproducibility
################################################################################
def fix_seeds(
    os=None, random=None, np=None, torch=None, tf=None, seed=42, verbose=True
):
    os_str = "os" if os is not None else ""
    random_str = "random" if random is not None else ""
    np_str = "np" if np is not None else ""
    torch_str = "torch" if torch is not None else ""
    tf_str = "tf" if tf is not None else ""

    # https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
    if os is not None:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)

    if random is not None:
        random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

    if tf is not None:
        tf.random.set_seed(seed)

    if verbose:
        print(f"\n{'-'*40}\n")
        print(
            f"Random seeds of the following packages have been fixed as {seed}"
        )
        print(os_str, random_str, np_str, torch_str, tf_str)
        print(f"\n{'-'*40}\n")


def gen_ID(time_format="%YY-%mM-%dD-%Hh%Mm%Ss", N=8):
    import random
    import string
    from datetime import datetime

    now = datetime.now()
    # now_str = now.strftime("%Y-%m-%d-%H-%M")
    now_str = now.strftime(time_format)

    # today_str = now.strftime("%Y-%m%d")
    randlst = [
        random.choice(string.ascii_letters + string.digits) for i in range(N)
    ]
    rand_str = "".join(randlst)
    return now_str + "_" + rand_str


def gen_timestamp():
    from datetime import datetime

    now = datetime.now()
    now_str = now.strftime("%Y-%m%d-%H%M")
    return now_str
