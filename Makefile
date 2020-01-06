build:
	docker build . -f ./Dockerfile -t privacy_and_ml:v0.1

run:
	DATA_DIRECTORY=${1:-$(PWD)/data}

	docker run \
	--name container_privacy_and_ml \
	-d \
	-v $(PWD)/workspace:/workspace \
	-p 8888:8888 privacy_and_ml:v0.1

logs:
	docker logs container_privacy_and_ml  2>&1 | grep "token"

stop_and_rm:
	docker stop container_privacy_and_ml
	docker rm container_privacy_and_ml