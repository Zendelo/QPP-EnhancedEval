import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qpptk",
    version="0.0.1",
    scripts=['qpptk/load_text_index.py', 'qpptk/utility_functions.py', 'qpptk/retrieval_local_manager.py',
             'qpptk/global_manager.py', 'qpptk/parse_queries.py'],
    author="Oleg Zendel",
    author_email="oleg.zendel@rmit.edu.au",
    description="QPP framework package",
    # long_description="Proof of concept of QPP framework development in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy', 'pandas', 'toml', 'protobuf', 'lmdb', 'lxml', 'syct'],
)
