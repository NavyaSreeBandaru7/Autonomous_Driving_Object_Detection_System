"""
Setup script for Autonomous Driving Object Detection System
==========================================================
Professional package configuration for distribution and installation
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements from requirements.txt
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'opencv-python>=4.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'Pillow>=8.3.0',
        'ultralytics>=8.0.0',
        'PyYAML>=5.4.0',
        'seaborn>=0.11.0',
        'streamlit>=1.15.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
        'tqdm>=4.62.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'requests>=2.26.0'
    ]

# Development requirements
dev_requirements = [
    'pytest>=6.2.0',
    'pytest-cov>=3.0.0',
    'black>=21.9.0',
    'flake8>=4.0.0',
    'pre-commit>=2.15.0',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=1.0.0'
]

setup(
    name="autonomous-driving-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time object detection system for autonomous vehicles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YourUsername/autonomous-driving-detection",
    project_urls={
        "Bug Tracker": "https://github.com/YourUsername/autonomous-driving-detection/issues",
        "Documentation": "https://github.com/YourUsername/autonomous-driving-detection/wiki",
        "Source Code": "https://github.com/YourUsername/autonomous-driving-detection",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": ["torch>=1.10.0+cu118", "torchvision>=0.11.0+cu118"],
        "api": ["flask>=2.0.0", "flask-cors>=3.0.0"],
        "web": ["streamlit>=1.15.0"],
        "all": dev_requirements + ["flask>=2.0.0", "flask-cors>=3.0.0", "streamlit>=1.15.0"]
    },
    entry_points={
        "console_scripts": [
            "autonomous-detect=main:main",
            "autonomous-api=flask_app:main",
            "autonomous-demo=streamlit_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md", "*.txt"],
        "data": ["*.jpg", "*.png", "*.mp4"],
        "docs": ["*.md", "*.rst"],
    },
    zip_safe=False,
    keywords=[
        "autonomous driving",
        "object detection",
        "computer vision",
        "machine learning",
        "yolo",
        "pytorch",
        "opencv",
        "real-time",
        "automotive",
        "ai"
    ],
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    tests_require=dev_requirements,
)

# Custom installation messages
print("""
🚗 Autonomous Driving Object Detection System
============================================

Installation completed successfully! 

📋 Next Steps:
1. Test installation: python -c "import main; print('✅ Installation verified')"
2. Run demo: python main.py --input 0
3. Start web interface: streamlit run streamlit_demo.py
4. Launch API server: python flask_app.py

📚 Documentation: https://github.com/YourUsername/autonomous-driving-detection

🎯 Quick Start Commands:
• autonomous-detect --input data/test.jpg     # Process image
• autonomous-detect --input 0                 # Use webcam
• autonomous-api                               # Start REST API
• autonomous-demo                              # Launch web demo

💡 For GPU acceleration, install CUDA-enabled PyTorch:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

🔧 Development mode:
   pip install -e .[dev]

Happy detecting! 🎉
""")
