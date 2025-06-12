Installation
============

Requirements
------------

MNGS requires Python 3.8 or later. The package has been tested on Linux, macOS, and Windows.

Basic Installation
------------------

Install MNGS using pip:

.. code-block:: bash

   pip install mngs

Development Installation
------------------------

For development or to get the latest features, install from the GitHub repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ywatanabe1989/mngs.git
   cd mngs

   # Install in development mode
   pip install -e .

Optional Dependencies
---------------------

MNGS has optional dependencies for specific features:

**For GPU support (PyTorch)**:

.. code-block:: bash

   pip install torch torchvision

**For advanced signal processing**:

.. code-block:: bash

   pip install scipy scikit-learn

**For database functionality**:

.. code-block:: bash

   pip install psycopg2-binary  # PostgreSQL
   pip install pymongo          # MongoDB

**For web scraping**:

.. code-block:: bash

   pip install beautifulsoup4 requests

Complete Installation
---------------------

To install all optional dependencies:

.. code-block:: bash

   pip install mngs[all]

Verifying Installation
----------------------

To verify that MNGS is installed correctly:

.. code-block:: python

   import mngs
   print(mngs.__version__)

You should see the version number printed without any errors.

Troubleshooting
---------------

**Import Errors**: If you encounter import errors, ensure all required dependencies are installed:

.. code-block:: bash

   pip install -r requirements.txt

**Permission Errors**: On Linux/macOS, you may need to use sudo or install in a virtual environment:

.. code-block:: bash

   # Using virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install mngs

**GPU Not Detected**: For GPU support, ensure CUDA is properly installed and PyTorch is configured for GPU:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())  # Should return True