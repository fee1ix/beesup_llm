from setuptools import setup, find_packages

setup(
    name="beesup_llm",
    version="0.1",
    packages=find_packages(),   # This will find all nested submodules
    install_requires=[
        'anytree', #2.12.1
        'datasets', #3.2.0
        'ipython', #8.12.3
        'kneed', #0.8.5
        'matplotlib', #3.8.3
        'numpy', #2.2.4
        'pandas', #2.2.3
        'peft', #0.10.0
        'plotly', #5.24.1
        'pydantic', #2.10.6
        'pytz', #2024.1
        'PyYAML', #6.0.1
        'rapidfuzz', #3.8.1
        'regex', #2023.12.25
        'scikit_learn', #1.6.1
        'scipy', #1.15.2
        'seaborn', #0.13.2
        'torch', #2.3.1
        'transformers', #4.48.3
        'trl', #0.14.0
    ]
)
