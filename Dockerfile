FROM ubuntu:latest
LABEL authors="mahar"

ENTRYPOINT ["top", "-b"]