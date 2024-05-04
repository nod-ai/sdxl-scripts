# Build with `docker build . -t sdxl-turbine --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) ./ubuntu_rocm_sdxl_python.dockerfile`
# Run with `docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add $(getent group render | cut -d: -f3) --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /path/to/downloaded/sdxl/weights:/weights -v /path/to/generated_imgs/:/gen_imgs sdxl-turbine'
# To benchmark inside docker: `./benchmark-txt2img.sh N /weights`
# To run python txt2img: `./run-txt2img-python.sh <prompt> <negative_prompt> <batch_count> <device_num> /weights`

FROM rocm/dev-ubuntu-22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Basic development environment
RUN apt-get update && apt-get install -y \
  software-properties-common git \
  build-essential cmake ninja-build clang lld vim python3.10-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
  pip install pybind11 nanobind numpy

# Checkout and build IREE
RUN git clone --depth=1 https://github.com/iree-org/iree.git && \
  cd iree && git submodule update --init --depth=1
RUN cd iree && cmake -S . -B build-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_EXTERNAL_HAL_DRIVERS="rocm" \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(which python3)" && \
  cmake --build build-release/ --target tools/all && \
  cmake --build build-release/ --target install

# Make IREE tools discoverable in PATH
ENV PATH=/iree/build-release/tools:$PATH
ENV PYTHONPATH=/iree/build-release/runtime/bindings/python:/iree/build-release/compiler/bindings/python

ARG ROCM_CHIP=gfx942
# Check out SDXL scripts and build model
RUN git clone --depth=1 https://github.com/monorimet/sdxl-scripts && cd sdxl-scripts && ./compile-txt2img.sh ${ROCM_CHIP}

RUN git clone https://github.com/iree-org/iree-turbine && \
  cd iree-turbine && \
  pip install -r requirements.txt .
  
RUN git clone https://github.com/nod-ai/SHARK-Turbine -b ean-sdxl-fixes && \
  cd SHARK-Turbine && \
  pip install --pre --upgrade -e models -r models/requirements.txt && \
  pip uninstall iree-compiler iree-runtime -y

WORKDIR /sdxl-scripts
ENTRYPOINT /bin/bash
