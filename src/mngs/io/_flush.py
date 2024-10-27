import sys
import os

def flush(sys=sys):
    """
    Flushes the system's stdout and stderr, and syncs the file system.
    This ensures all pending write operations are completed.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os.sync()
