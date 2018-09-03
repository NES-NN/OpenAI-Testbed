# OpenAI Testbed

[![Build Status][travis-image]][travis] [![MIT License][license-image]][license]

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

__NOTE__: On creating the container the local `train` directory is shared within the container _but_ any edits to the scripts contained within will not be propogated to Wooey until a relaunch of the container.

### Using the scripts

Please see the [wiki](https://github.com/NES-NN/OpenAI-Testbed/wiki) for information on how to use the Training and Utility scripts.

## Copyright and license

Copyright (c) 2018 Joshua Beemster, Jasim Schluter, Loic Nyssen 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[travis-image]: https://travis-ci.org/NES-NN/OpenAI-Testbed.svg?branch=master
[travis]: https://travis-ci.org/NES-NN/OpenAI-Testbed

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
[license]: https://opensource.org/licenses/MIT

[gym-super-mario]: https://github.com/ppaquette/gym-super-mario
[gym-author]: https://github.com/ppaquette
[wooey]: https://github.com/wooey/Wooey
[docker]: https://docs.docker.com/install/
