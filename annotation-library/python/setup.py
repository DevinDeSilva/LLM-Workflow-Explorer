from setuptools import setup, find_packages

setup(
    name="workflow_annotator",
    version="0.1.0",
    description="Explanation Annotator used to create a KG from provenance execution traces",
    author="Devin De Silva",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "rdflib",
        "pyyaml",
    ],
    python_requires=">=3.6",
)
