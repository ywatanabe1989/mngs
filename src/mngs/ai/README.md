# [`mngs.ai`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/ai/)

## Overview
The `mngs.ai` module provides a collection of artificial intelligence and machine learning utilities, focusing on deep learning with PyTorch and various AI-related tasks.

## Installation
```bash
pip install mngs
```

## Features
- Classification utilities (ClassificationReporter, ClassifierServer)
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
reporter = ai.ClassificationReporter(model)

# Early stopping
early_stopping = ai.EarlyStopping(patience=10)

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
- `mngs.ai.ClassificationReporter`: Utility for classification model evaluation
- `mngs.ai.ClassifierServer`: Server for deploying classification models
- `mngs.ai.EarlyStopping`: Implementation of early stopping for training
- `mngs.ai.LearningCurveLogger`: Logger for tracking and visualizing learning curves
- `mngs.ai.clustering.UMAP`: UMAP implementation for clustering
- `mngs.ai.layer`: Custom PyTorch layers
- `mngs.ai.loss`: Custom loss functions
- `mngs.ai.optim`: Optimization utilities including Ranger optimizer
- `mngs.ai.metrics`: Various evaluation metrics
- `mngs.ai.genai`: Interfaces for generative AI models

## Use Cases
- Deep learning model development and training
- Model evaluation and performance analysis
- Clustering and dimensionality reduction
- Natural language processing tasks
- Generative AI applications

## Performance
The `mngs.ai` module is built on top of PyTorch, leveraging GPU acceleration for computationally intensive tasks when available.

## Contributing
Contributions to improve `mngs.ai` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).

