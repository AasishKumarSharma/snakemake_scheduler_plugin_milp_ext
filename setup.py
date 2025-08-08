from setuptools import setup, find_packages

setup(
    name="snakemake-scheduler-plugin-milp-ext",
    version="0.1.0",
    author="Aasish Kumar Sharma",
    author_email="aasish.sharma@uni-goettingen.de", 
    description="MILP-based job scheduler plugin for Snakemake",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "snakemake-interface-scheduler-plugins>=2.0.0,<3.0.0",
        "pulp>=2.0",
        "networkx>=2.5",
        "pyyaml>=5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
        ],
    },
    entry_points={
        "snakemake.scheduler_plugins": [
            "milp-ext = snakemake_scheduler_plugin_milp_ext:Scheduler",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
