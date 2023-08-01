# Docker image with qpptk

Build the docker image via this command:

```
docker build -t qpptk .. -f ./Dockerfile
```

As soon as we have finalized everything and we have published the docker image, the build command above can be removed because others can directly use the published image.

If the image was build, you can run qpptk via Docker with the following command (e.g., printing the help message):

```
docker run --rm -ti qpptk --help
```

