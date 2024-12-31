from setuptools import setup, find_packages

setup(
    name='ai_server',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'rich',
        'fastapi',
        'uvicorn',
        'requests',
        'python-dotenv',
        'langchain-core',
        'langchain-openai',
        'kimiconfig',
        'caldav',
        'urllib3',
        # Добавьте другие зависимости, которые используются в вашем проекте
    ],
    entry_points={
        'console_scripts': [
            'ai_server=ai_server.main:main',
        ],
    },
    author='kimifish',
    author_email='kimifish@proton.me',
    description='AI Server using LangChain and FastAPI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ai_server',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)