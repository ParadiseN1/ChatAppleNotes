from setuptools import setup, find_packages

setup(
    name='chat_notes',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'macnotesapp',
        'ollama',
        'termcolor',
        'transformers',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'chat-notes=src.parse_notes_local:cli',
        ],
    },
)