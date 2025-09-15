# Define the image name
IMAGE_NAME = mini_data_analysis
DOCKER_ID_USER = aeshagandhi27

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 main.py

all: 
	install format lint test

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
    docker run -p 5004:5000 $(IMAGE_NAME)

test:
	pytest -v test_main.py

# Remove the Docker image
clean:
	docker rmi $(IMAGE_NAME)

image_show:
	docker images

container_show:
	docker ps -a

push:
	docker login
	docker tag $(IMAGE_NAME) $(DOCKER_ID_USER)/$(IMAGE_NAME)
	docker push $(DOCKER_ID_USER)/$(IMAGE_NAME):latest

login:
	docker login -u ${DOCKER_ID_USER}