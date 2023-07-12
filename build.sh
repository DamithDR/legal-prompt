docker rmi legal-prompt:latest
docker build -f DockerFile -t legal-prompt:latest .
docker run legal-prompt:latest