from distutils.core import setup

setup(
    name='generative_rf',
    version='1.0',
    description='Generative Random Forest for online learning',
    author='Romain Mouret',
    author_email='mouret.romain@gmail.com',
    url='https://github.com/rom1mouret/generative_rf',
    packages=['generative_rf'],
    install_requires=["scikit-learn>=0.17.0", "numpy>=1.8.0"]
)
