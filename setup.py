from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Financial and Insurance Industry',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='risk-modeling-tools',
    version='0.0.2',
    description='Tools for credit risk scoring models development and portfolio management',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mgrts/risk_modeling_tools',
    author='Mikhail Gritskikh',
    author_email='m.gritskikh@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords=['credit', 'risk', 'modeling', 'scoring', 'binning'],
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'sklearn', 'scipy']
)
