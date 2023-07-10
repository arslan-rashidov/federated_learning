# federated_learning

# Conda 
conda create --name federated_learning
conda activate federated_learning

# Install federated_learning
git clone https://github.com/arslan-rashidov/federated_learning
cd federated_learning/
pip install -e .

# Try MNIST Federated Learning Example
cd federated_learning/examples/mnist_fl/server/
python3 server.py

To monitor federated learning process:
mlflow ui --port 8888
...and open http://127.0.0.1:8888
