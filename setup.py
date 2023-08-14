import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="flyxdm",
    version="0.0.0",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com",
    description="Python primitives for - On the fly far field X-ray Diffraction crystal state estimation.",
    long_description=long_description,
    long_description_content_type="text/md",
    url="https://github.com/AxelHenningsson/flyxdm",
    project_urls={
        "Documentation": "https://axelhenningsson.github.io/flyxdm/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.2,<3.10",
    install_requires=['numpy', 'scipy', 'astra-toolbox', 'matplotlib', 'xfab']
)
