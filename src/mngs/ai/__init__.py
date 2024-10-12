#!/usr/bin/env python3

try:
    from . import act, clustering, layer, metrics, optim, plt, sk, utils
except ImportError as e:
    print(f"Warning: Failed to import some modules.")

try:
    from ._gen_ai._genai_factory import genai_factory as GenAI
except ImportError as e:
    print(f"Warning: Failed to import GenAI.")

try:
    from .ClassificationReporter import (
        ClassificationReporter,
        MultiClassificationReporter,
    )
except ImportError as e:
    print(f"Warning: Failed to import ClassificationReporter.")

try:
    from .ClassifierServer import ClassifierServer
except ImportError as e:
    print(f"Warning: Failed to import ClassifierServer.")

try:
    from .EarlyStopping import EarlyStopping
except ImportError as e:
    print(f"Warning: Failed to import EarlyStopping.")

try:
    from .LearningCurveLogger import LearningCurveLogger
except ImportError as e:
    print(f"Warning: Failed to import LearningCurveLogger.")

try:
    from .loss.MultiTaskLoss import MultiTaskLoss
except ImportError as e:
    print(f"Warning: Failed to import MultiTaskLoss.")

# #!/usr/bin/env python3

# from . import act, clustering, layer, metrics, optim, plt, sk, utils
# from ._gen_ai._genai_factory import genai_factory as GenAI
# from .ClassificationReporter import (
#     ClassificationReporter,
#     MultiClassificationReporter,
# )
# from .ClassifierServer import ClassifierServer
# from .EarlyStopping import EarlyStopping
# from .LearningCurveLogger import LearningCurveLogger

# # from ._switchers import switch_layer, switch_act, switch_optim
# from .loss.MultiTaskLoss import MultiTaskLoss
