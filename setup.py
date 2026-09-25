from setuptools import setup, find_packages

setup(
    name="transformerlens-align",
    version="1.0.0",
    description="Extension of TransformerLens for Sycophancy Circuits, Causal Activation Patching, and Closed-Loop Alignment Interventions in Autoregressive LLMs.",
    author="AI Safety & Mechanistic Interpretability Research Group",
    author_email="research@alignment-manifesto.org",
    url="https://github.com/svdgamerz/TransformerLens-Align",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformer-lens>=1.14.0",
        "transformers>=4.30.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
)
