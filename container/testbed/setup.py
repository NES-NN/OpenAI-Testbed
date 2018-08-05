from setuptools import setup, find_packages
import os


version_file_path = os.path.join(
    os.path.dirname(__file__),
    'testbed',
    'version.py'
    )
exec(open(version_file_path).read(), {}, locals())


setup(
    name="testbed",
    version=__version__,
    description="Common library for OpenAI Training Scripts",
    author="Joshua Beemster",
    author_email="joshua.a.beemster@gmail.com",
    url="https://github.com/NES-NN/OpenAI-Testbed",
    download_url="https://github.com/NES-NN/OpenAI-Testbed/tarball/%s" % __version__,
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "neat-python",
        "argparse",
        "colour",
        "click",
        "pandas==0.22.0"
    ],
    tests_require=[
        "nose",
        "coverage",
        "pylint"
    ],
    test_suite="nose.collector"
)
