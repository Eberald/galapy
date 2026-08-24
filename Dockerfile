# Author: Enrico Veraldi
# Dockerfile Cloudy+galapy+CloudIA

############################ Stage 1 — toolchain ##############################
# Ubuntu 22.04
FROM ubuntu:22.04 AS toolchain

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make perl \
        wget ca-certificates git \
        python3 python3-pip python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

########################## Stage 2 — build CLOUDY #############################
FROM toolchain AS cloudy-build

ARG CLOUDY_URL=https://data.nublado.org/cloudy_releases/c25/c25.00.tar.gz
ARG CLOUDY_SHA256=12a4fac7a29f888f56b37885df0067c92eae47a53c78060743e3c92140add5f3

WORKDIR /opt
RUN [ "${CLOUDY_SHA256}" != "TO FIX" ] || { \
      echo "ERRORE: Fix the source:"; \
      echo "  wget -q ${CLOUDY_URL} && sha256sum $(basename ${CLOUDY_URL})"; \
      echo "  docker build --build-arg CLOUDY_SHA256=<valore> ..."; \
      exit 1; } \
 && wget -q "${CLOUDY_URL}" -O cloudy.tar.gz \
 && echo "${CLOUDY_SHA256}  cloudy.tar.gz" | sha256sum -c - \
 && mkdir cloudy \
 && tar xzf cloudy.tar.gz -C cloudy --strip-components=1 \
 && rm cloudy.tar.gz

#workdir cloudy source
WORKDIR /opt/cloudy/source


RUN make cloudy.exe -j"$(nproc)" \
      OPT='-O3 -ftrapping-math -fno-math-errno -fasynchronous-unwind-tables -Wno-deprecated-declarations' \
 && sha256sum cloudy.exe | tee /opt/cloudy/cloudy.exe.sha256

ENV CLOUDY_DATA_PATH=/opt/cloudy/data

# smoke test Cloudy
RUN mkdir -p /tmp/smoke && cd /tmp/smoke \
 && printf 'test\n' | /opt/cloudy/source/cloudy.exe > smoke.out 2>&1 \
 && grep -q 'Cloudy exited OK' smoke.out \
 && grep -m1 'Cloudy 2' smoke.out \
 && rm -rf /tmp/smoke

################### Stage 3 — Python 3.10 + galapy/cloudia ###################
FROM toolchain AS pystack

WORKDIR /opt/galapy

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir pybind11

COPY galapy/spectroscopy/requirements.txt ./galapy/spectroscopy/requirements.txt
RUN python3 -m pip install --no-cache-dir -r galapy/spectroscopy/requirements.txt

COPY . /opt/galapy
RUN python3 -m pip install --no-cache-dir -e ".[cloudia]" \
 && python3 -c "import galapy, galapy.spectroscopy, torch, gpytorch, sklearn, yaml; \
print('stack OK | galapy', galapy.__version__, '| torch', torch.__version__, '| gpytorch', gpytorch.__version__)"
######################### Stage 4 — runtime ###########################
FROM pystack AS runtime

COPY --from=cloudy-build /opt/cloudy /opt/cloudy

ENV CLOUDY_EXE=/opt/cloudy/source/cloudy.exe \
    CLOUDY_DATA_PATH=/opt/cloudy/data \
    CLOUDIA_WORKFLOWS=/opt/galapy/workflow \
    CLOUDIA_UTILS=/opt/galapy/galapy/spectroscopy/utils \
    GALAPY_DATASET=/opt/galapy_dataset \
    CLOUDIA_CONFIGS=/opt/galapy_dataset/nebular/configs \
    PATH=/opt/cloudy/source:${PATH}

RUN ln -s /opt/cloudy/source/cloudy.exe /usr/local/bin/cloudy \
 && test -x /usr/local/bin/cloudy \
 && python3 -c "import galapy, galapy.spectroscopy; print('import ok')" \
 && python3 -c "from galapy.spectroscopy.utils import require_cloudy; \
d = require_cloudy(check_version=True); \
print('CLOUDY ok |', d.exe, '|', d.version)"

WORKDIR /work
CMD ["bash"]

LABEL org.opencontainers.image.title="cloudia" \
      org.opencontainers.image.description="CLOUDY C25.00 + fork GalaPy [cloudia]" \
      org.opencontainers.image.licenses="GPL-3.0-only AND Zlib" \
      org.opencontainers.image.source="https://github.com/Eberald/galapy"