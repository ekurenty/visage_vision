from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='visage_vision',
      version="0.0.1",
      description="Emotions Detections Model (api_pred)",
      license="LeWagon",
      author="1577 Melbourne",
      author_email="contact@lewagon.org",
      url="https://github.com/ekurenty/visage_vision.git",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)