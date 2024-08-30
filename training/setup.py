from setuptools import find_packages, setup

setup(
    name="train_isolation_forest",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "google-cloud-aiplatform==1.64.0",
        # "google-cloud-storage==2.18.2",
        "google-cloud-datastore",
        "google-cloud-logging",
        # TODO: Add additional required packages
        "google-api-python-client",
        "googledatastore",
        "argparse",
        "bson==0.5.10",
        "numpy==1.26.0",
        "onnx==1.16.2",
        "onnxconverter-common @ git+https://github.com/microsoft/onnxconverter-common@209a47e18e6a4c3474273a0b2a5e8f1fda481643",
        "pandas",
        "python-dateutil",
        "python-dotenv",
        "skl2onnx",
        "viam-sdk==0.29.1",
    ],
)
