from setuptools import find_packages, setup

setup(
    name="train_if",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "google-cloud-aiplatform==1.64.0",
        "google-cloud-datastore",
        # TODO: Add additional required packages
        # verify
        "skl2onnx",
        # "onnxconverter-common@git+https://github.com/microsoft/onnxconverter-common@209a47e18e6a4c3474273a0b2a5e8f1fda481643",
        "bson",
        "python-dotenv",
        "viam-sdk==0.29.1",
        "google-cloud-logging",
        "onnx==1.16.2",
    ],
)
