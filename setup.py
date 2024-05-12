from setuptools import setup
if __name__ == '__main__':

    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    setup(install_requires=requirements)