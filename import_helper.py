"""
Import Helper for ProQuants Professional
Ensures proper import paths for all modules
"""

import sys
import os

def setup_imports():
    """Setup import paths for ProQuants modules"""
    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add src directory to path
    src_dir = os.path.join(root_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Add root directory to path
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    return True

# Auto-setup when imported
setup_imports()
