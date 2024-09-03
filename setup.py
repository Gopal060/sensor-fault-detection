from setuptools import find_packages, setup
from typing import List  


def get_requirements() -> List[str]:   # type: ignore
    """This get_requirements func returns list of strings form requirements.txt file."""
    requirements: List[str] = []

    
    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    return requirements
        
    


setup(
    name="sensor",
    version="0.0.1",
    author="Gopal",
    author_email="gopalrkate@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements(),  # This install_requires wants list of string to perform.
)
