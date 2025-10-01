from setuptools import setup, find_packages

# Lê dependências do requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RailroadDES",  # 🔹 nome do seu pacote (pode escolher outro)
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "scripts*"]),  # encontra automaticamente os pacotes
    install_requires=requirements,
    python_requires=">=3.11",
)
