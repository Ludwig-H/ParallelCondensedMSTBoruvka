
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np, os

ext = Extension(
    "parallel_condensed_mst_boruvka._boruvka",
    [os.path.join("src","parallel_condensed_mst_boruvka","_boruvka.pyx")],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3","-march=native"],
)
setup(
    name="ParallelCondensedMSTBoruvka",
    version="0.4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize([ext], language_level=3, compiler_directives={
        "boundscheck": False, "wraparound": False, "cdivision": True, "initializedcheck": False
    }),
    include_package_data=True,
    install_requires=["numpy>=1.22","joblib>=1.2"],
    python_requires=">=3.9",
)
