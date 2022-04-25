Build the project using the standard CMake workflow:

```
mkdir build
cd build
cmake ..
make
```

Don't forget to set the build configuration to Release mode for a dramatic increase in performance.

# Dependencies
This project needs the libigl and polyscope libraries.
 * libigl: https://github.com/libigl/libigl
 * polyscope: https://polyscope.run/

Libigl should get installed automatically. You will need to download the Polyscope library yourself, and set the `POLYSCOPE_DIR` environment variable to the path where you installed Polyscope.

