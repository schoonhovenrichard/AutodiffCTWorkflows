from setuptools import setup, find_packages

setup(
    name = 'autodiff_ct_workflows',
    packages=find_packages(),
    version = '0.1',  # Ideally should be same as your GitHub release tag varsion
    description = 'Package accompanying auto-differentiation for CT workflows paper',
    author = 'Richard Schoonhoven, Alexander Skorikov',
    author_email = 'r.a.schoonhoven@hotmail.com',
    url = 'https://github.com/schoonhovenrichard/AutodiffCTWorkflows',
    keywords = ["auto-tuning","pipeline","ct","computed tomography","pytorch","beam hardening","phase retrieval"],
    classifiers = [],
    install_requires=[],
)
