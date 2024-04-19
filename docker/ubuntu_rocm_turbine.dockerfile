# Build with `docker build . -t sdxl-repro --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) -f ./ubuntu_rocm_turbine.dockerfile`
# Run with `docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add $(getent group render | cut -d: -f3)
# --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /path/to/downloaded/sdxl/weights:/weights sdxl-repro`
# To benchmark inside docker: `./benchmark-txt2img.sh N /weights`

FROM rocm/dev-ubuntu-22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Basic development environment
RUN apt-get update && apt-get install -y \
  software-properties-common git \
  build-essential cmake ninja-build clang lld && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up mirror user account
ARG DOCKER_USERID=0
ARG DOCKER_GROUPID=0
ARG DOCKER_USERNAME=mirror
ARG DOCKER_GROUPNAME=mirror
RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_GROUPID} -ne 0 ]; then \
    groupadd --gid ${DOCKER_GROUPID} ${DOCKER_GROUPNAME} && \
    useradd --no-log-init --create-home \
      --uid ${DOCKER_USERID} --gid ${DOCKER_GROUPID} \
      --shell /usr/bin/zsh ${DOCKER_USERNAME}; \
fi

# Now switch to the mirror user home directory
USER ${DOCKER_USERNAME}
WORKDIR /home/${DOCKER_USERNAME}
run rm -rf ./iree

# Checkout and build IREE
RUN git clone --depth=1 https://github.com/openxla/iree.git -b sdxl && \
  cd iree && git submodule update --init --depth=1
RUN cd iree && cmake -S . -B build-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_EXTERNAL_HAL_DRIVERS="rocm" && \
  cmake --build build-release/ --target tools/all

# Make IREE tools discoverable in PATH
ENV PATH=/home/${DOCKER_USERNAME}/iree/build-release/tools:$PATH

# Check out SDXL scripts and build model
RUN git clone --depth=1 https://github.com/monorimet/sdxl-scripts -b gfx942-iree-main
RUN cd sdxl-scripts && ./compile-txt2img.sh

WORKDIR /home/${DOCKER_USERNAME}/sdxl-scripts
ENTRYPOINT /bin/bash