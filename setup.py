from setuptools import setup, find_packages

setup(
    name='quantitative_framework',
    version='0.1.0',
    description='A Python framework for quantitative trading strategy backtesting and analysis.',
    author='Your Name',  # 您可以稍後修改為您的名字
    author_email='your.email@example.com', # 以及您的郵箱
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'ta',
        'numba',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # 假設為 MIT 授權
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
