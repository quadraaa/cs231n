
docker build -t cs231n-py3.7 .
docker run -v $(pwd):/src/notebooks -p 8888:8888 -td cs231n-pytorch-cpu