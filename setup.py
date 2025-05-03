from setuptools import setup, find_packages

setup(
    name='mentalhealth-nlp',
    version='0.1.0',
    description='Detect and classify mental health-related social media posts using NLP techniques for early intervention and awareness.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Keshav Elango',
    author_email='keshavelangousa@gmail.com',
    url='https://github.com/KeshavElangoDS/Mental-Health-Analysis-using-Social-Media-Data',
    license='MIT',
    packages=find_packages(include=['mentalhealth', 'mentalhealth.*']),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Streamlit',
    ],
    python_requires='>=3.7',
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'run-app=mentalhealth.app:main',  # if you have a main() in app.py
        ],
    },
)
