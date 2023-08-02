# federated_learning

# Conda 
```bash
conda create --name federated_learning
conda activate federated_learning
```

# Install federated_learning
```bash
git clone https://github.com/arslan-rashidov/federated_learning
cd federated_learning/
pip install -e .
```

# How to run federated learning examples
```bash
cd federated_learning/examples/EXAMPLE_NAME/server/ (e.g. cd federated_learning/examples/mnist_fl/server/)
python3 server.py
```

To monitor federated learning process:
```bash
mlflow ui --port 8888
```
...and open http://127.0.0.1:8888
