from setuptools import setup, find_packages

setup(
    name='federated_learning',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'mlflow==2.3.1',
        'numpy==1.24.3',
        'scikit_learn==1.2.2',
        'setuptools==67.8.0'
    ]
)

