from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> list[str]:
    '''
    This Function will return the list of requirements
    '''
    requirements = []
    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Parasuram',
    author_email='parasuramgoud30@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)