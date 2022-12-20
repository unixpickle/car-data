from setuptools import setup

setup(
    name="car-data",
    version="0.0.1",
    description="Train a car price estimator.",
    packages=["car_data"],
    install_requires=[
        "torch",
        "torchvision",
        "sk2torch",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    author="Alex Nichol",
    author_email="unixpickle@gmail.com",
    url="https://github.com/unixpickle/car-data",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
