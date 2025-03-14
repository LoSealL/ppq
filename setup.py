from setuptools import find_packages, setup

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    vars = {}
    with open("ppq/core/config.py", encoding="utf-8") as f:
        code = compile(f.read(), "ppq/core/config.py", mode="exec")
    # python 3.12 and before, exec signature was:
    # exec(source, globals, locals, /, *, closure=None)
    # While python 3.13 signature becomes:
    # exec(source, /, globals, locals, *, closure=None)
    exec(code, None, vars)
    return vars["PPQ_CONFIG"].VERSION


setup(author='ppq',
      author_email='dcp-ppq@sensetime.com',
      description='PPQ is an offline quantization tools',
      long_description=readme(),
      long_description_content_type='text/markdown',
      install_requires=open('requirements.txt').readlines(),
      python_requires='>=3.6',
      name='ppq',
      packages=find_packages(),
      classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
      license='Apache License 2.0',
      include_package_data=True,
      version=get_version(),
      zip_safe=False
    )
