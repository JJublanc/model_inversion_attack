# Privacy and machine learning

This project aim to reproduce an model inversion attack against a deep classifier of hand written digits (mnist).
Reference paper : https://arxiv.org/abs/1911.07135

Before starting make sure you have "MAKE" install (if not you can launch directly the docker commands)

Launch the container with the following commands in terminal :

1 Build the Docker image
```python
make build
```

2 Run the Docker image
```python
make run
```

3 Return logs to get the address of the jupyter lab page (with the token needed)
```python
make logs
```

When finished you can close and remove the container
```python
make stop_and_rm
```
