from setuptools import setup

setup(
    name="PrivateModelArchitectures",
    version="0.1",
    description="Implementations of torch model architectures that have been proven to work well with DP training",
    author="AIMLAB@TU Munich",
    author_email="alex.ziller@tum.de",
    packages=[
        "PrivateModelArchitectures",
        "PrivateModelArchitectures.classification",
        "PrivateModelArchitectures.segmentation",
    ],
    url="https://github.com/TUM-AIMED/PrivateModelArchitectures",
)
