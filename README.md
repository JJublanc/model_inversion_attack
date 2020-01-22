# Privacy and machine learning

This project aims at reproducing a model inversion attack against a deep classifier of hand written digits (mnist).

Reference paper: https://arxiv.org/abs/1911.07135

Before starting make sure you have `make` installed (if not you can launch directly the docker commands)

Launch container with the following commands in terminal:

1. Build the Docker image
```
make build
```

2. Run the Docker image
```
make run
```

3. Return logs to get the address of the jupyter lab page (with the token needed)
```
make logs
```

4. When finished you can close and remove the container
```
make stop_and_rm
```
