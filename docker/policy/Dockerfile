# A local copy of the scoring server's policy container: the image your agent.py runs in, one container per episode.
#
# Same base image (by digest), same pinned packages (requirements.txt: numpy 2.4.5, SciPy 1.18.1, CPU torch 2.14.0 and
# what torch needs, each by hash) and same entry point as the organisers' image. The agent kit (the shim that talks to
# the scorer and turns its messages into your observation dict) comes from the vendored shockbench-flow wheel.
# `uv run sbf check <folder> --docker` builds it from a context of only this file,
# requirements.txt, launch.py and lib/ (the kit's packages unpacked from the wheel), then plays episodes in it.
# The torch pin is the linux/amd64 wheel, so the image is linux/amd64 (emulated on an ARM machine).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

# one thread for BLAS and torch, as in the scoring container (it runs with --cpus 1)
ENV UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
COPY requirements.txt /opt/sbf/requirements.txt
RUN uv pip install --system --no-deps --require-hashes --compile-bytecode -r /opt/sbf/requirements.txt \
    && python -I -c "import numpy, scipy, fastjsonschema, torch; \
assert numpy.__version__ == '2.4.5', numpy.__version__; \
assert scipy.__version__ == '1.18.1', scipy.__version__; \
assert torch.__version__ == '2.14.0+cpu', torch.__version__"
COPY launch.py /opt/sbf/launch.py
COPY lib/ /opt/sbf/lib/
RUN python -I -m compileall -q /opt/sbf/lib && chmod -R a+rX,go-w /opt/sbf \
    && python -I -B -c "import sys; sys.path.insert(0, '/opt/sbf/lib'); import shockbench_flow_agent as k; k.warm_up()"

ENV SBF_SHIM_MODULE=shockbench_flow_agent \
    SBF_KIT_PATH=/opt/sbf/lib \
    HOME=/tmp
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "-u", "/opt/sbf/launch.py"]
CMD ["/submission"]
