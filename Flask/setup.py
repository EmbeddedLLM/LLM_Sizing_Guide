from setuptools import setup, find_packages

setup(
    name='llmcalculator',  # Replace with your package's name
    version='0.1.0',  # Initial version
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,  # Include files specified in MANIFEST.in
    
    install_requires=[
        # List your dependencies here, e.g., 'flask'
        "Flask",
        "requests",
        "beautifulsoup4",
        "transformers",
        "tabulate"
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here if needed
            # 'command-name = module:function',
        ],
    },
    author='Tan Tun Jian',  # Replace with your name
    author_email='tunjian.tan@embeddedllm.com',  # Replace with your email
    description='This is a flask app for LLM Calculator',  # Short description
    # long_description=open('README.md').read(),  # Long description from README
    long_description_content_type='text/markdown',  # Content type for long description
    url='https://github.com/EmbeddedLLM/LLM_Sizing_Guide',  # URL to your project's repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version required
)