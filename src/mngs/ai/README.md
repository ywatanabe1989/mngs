<!-- ---
!-- title: README.md
!-- author: ywatanabe
!-- date: 2024-11-04 03:16:41
!-- --- -->

# [`mngs.ai`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/ai/)

## Overview
The `mngs.ai` module provides a collection of artificial intelligence and machine learning utilities, focusing on deep learning with PyTorch and various AI-related tasks.

## Installation
```bash
pip install mngs
```

## Features
- Classification utilities (ClassificationReporter, MultiClassificationReporter, ClassifierServer)
- Early stopping implementation for training
- Learning curve logging and visualization
- Clustering algorithms (UMAP)
- Custom PyTorch layers and loss functions
- Optimization utilities (Ranger optimizer)
- Metrics calculation and evaluation tools
- Generative AI interfaces (OpenAI, Claude, Llama, etc.)

## Quick Start
```python
import mngs.ai as ai
import torch
import torch.nn as nn

# Classification example
model = nn.Linear(10, 2)
reporter = ai.ClassificationReporter("./results")

# Early stopping
early_stopping = ai.EarlyStopping(patience=10, verbose=True)

# Learning curve logger
logger = ai.LearningCurveLogger()

# Custom loss function
loss_fn = ai.MultiTaskLoss()

# Optimization
optimizer = ai.optim.Ranger(model.parameters())

# Metrics
accuracy = ai.metrics.balanced_accuracy_score(y_true, y_pred)

# Generative AI
gen_ai = ai.genai.GenAI(model="gpt-3.5-turbo")
response = gen_ai.generate("Tell me a joke.")
```

## API Reference
- `mngs.ai.ClassificationReporter`: Utility for classification model evaluation and reporting
- `mngs.ai.MultiClassificationReporter`: Manages multiple ClassificationReporter instances for multi-target classification tasks
- `mngs.ai.ClassifierServer`: Server for deploying classification models
- `mngs.ai.EarlyStopping`: Implementation of early stopping for training
- `mngs.ai.LearningCurveLogger`: Logger for tracking and visualizing learning curves
- `mngs.ai.clustering.UMAP`: UMAP implementation for clustering
- `mngs.ai.layer`: Custom PyTorch layers
- `mngs.ai.loss`: Custom loss functions
- `mngs.ai.optim`: Optimization utilities including Ranger optimizer
- `mngs.ai.metrics`: Various evaluation metrics
- `mngs.ai.genai`: Interfaces for generative AI models

## Detailed Usage

### EarlyStopping

The `EarlyStopping` class is used to monitor the validation score during training and stop the process if no improvement is seen for a specified number of consecutive checks.

```python
from mngs.ai import EarlyStopping

early_stopping = EarlyStopping(patience=7, verbose=True, delta=1e-5, direction="minimize")
```

### ClassificationReporter

The `ClassificationReporter` class is used for reporting various classification metrics and saving them. It calculates and saves metrics such as Balanced Accuracy, Matthews Correlation Coefficient (MCC), Confusion Matrix, Classification Report, ROC AUC score/curve, and Precision-Recall AUC score/curve.

```python
from mngs.ai import ClassificationReporter

reporter = ClassificationReporter(save_directory)
reporter.calc_metrics(true_class, pred_class, pred_proba, labels)
reporter.summarize()
reporter.save()
```

### MultiClassificationReporter

The `MultiClassificationReporter` class manages multiple `ClassificationReporter` instances, one for each target in a multi-target classification task.

```python
from mngs.ai import MultiClassificationReporter

multi_reporter = MultiClassificationReporter(save_directory, targets=['target1', 'target2'])
multi_reporter.calc_metrics(true_class, pred_class, pred_proba, labels, tgt='target1')
multi_reporter.summarize(tgt='target1')
multi_reporter.save(tgt='target1')
```

## Use Cases
- Deep learning model development and training
- Model evaluation and performance analysis
- Clustering and dimensionality reduction
- Natural language processing tasks
- Generative AI applications
- Multi-target classification tasks

## Performance
The `mngs.ai` module is built on top of PyTorch, leveraging GPU acceleration for computationally intensive tasks when available.

## Contributing
Contributions to improve `mngs.ai` are welcome. Please submit pull requests or open issues on the GitHub repository.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
