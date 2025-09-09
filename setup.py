from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas","numpy","scikit-learn","imbalanced-learn",
        "matplotlib","xgboost","joblib"
    ],
)
