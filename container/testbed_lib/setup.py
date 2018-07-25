"""
    setup.py
"""


from setuptools import setup, find_packages
import os


version_file_path = os.path.join(
    os.path.dirname(__file__),
    'testbed_lib',
    'version.py'
    )
exec(open(version_file_path).read(), {}, locals())


setup(
    name="testbed-lib",
    version=__version__,
    description="Python SDK for all general purpose code to be used by the OpenAI Testbed",
    author="Joshua Beemster",
    author_email="joshua.a.beemster@gmail.com",
    url="https://github.com/NES-NN/OpenAI-Testbed",
    download_url="https://github.com/NES-NN/OpenAI-Testbed/tarball/%s" % __version__,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "graphviz",
        "matplotlib"
    ],
    tests_require=[
        "nose"
    ],
    test_suite="nose.collector"
)
