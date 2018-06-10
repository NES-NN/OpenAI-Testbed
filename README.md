# OpenAI Testbed

[![Build Status][travis-image]][travis]

## About

This project contains code and a test environment to train neural networks against Super Mario Bros for NES.  The project contains a forked version of the excellent [open source OpenAI Gym for Super Mario Bros][gym-super-mario] written by [Philip Paquette][gym-author].

The testbed works via the creation and launch of a Docker container which brings up several sub-systems:

1. A VNC server which lets you access a desktop environment for the container (accessed on port 5900)
2. A [Wooey][wooey] server which lets you launch headless training jobs within the container (accessed on port 8000)

Once launched you will be able to commence training your AI on Super Mario Bros!

### Quickstart

To use the testbed you will need to have a unix based system (tested on ubuntu & macOS) and [Docker][docker] installed.  Once these requirements are met you will need to run the following command from the root of this repository:

```
$ make up
```

This will build and launch the container ready for use.

To close the container:

```
$ make down
```

To reset the running container simply run `up` again - this will automatically close any existing container before re-launching:

```
$ make up
```

To rebuild the container:

```
$ make clear && make up
```

[travis-image]: https://travis-ci.org/NES-NN/OpenAI-Testbed.svg?branch=master
[travis]: https://travis-ci.org/NES-NN/OpenAI-Testbed
[gym-super-mario]: https://github.com/ppaquette/gym-super-mario
[gym-author]: https://github.com/ppaquette
[wooey]: https://github.com/wooey/Wooey
[docker]: https://docs.docker.com/install/
