from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='visage_vision',
      version="0.0.1",
      description="Facial Expression Detection Model",
      license="Le Wagon",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      url="https://github.com/ekurenty/visage_vision",
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
