from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Augmentare",
    version="0.0.1",
    description="A study on the fairness of generative augmentation methods for deep learning",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Vuong Nguyen",
    author_email="vuong.nguyen@irt-saintexupery.com",
    license="MIT",
    install_requires=['torch', 'torchvision', 'numpy', 'matplotlib', 'scipy', 'tqdm', 'imageio'],
    extras_require={
        "tests": ["pytest", "pytest-cov", "tox", "pylint"],
        "docs": ["mkdocs", "mkdocs-material", "numkdoc", "docutils", "Markdown", "mknotebooks", "Pygments", "pymdown-extensions"],
    },
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)