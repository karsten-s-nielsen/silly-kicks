# ADR-128 Cycle 1b (S5.12): the locked third-party environment only. No repo code, no data.
FROM python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0
ARG UV_REQUIREMENT
ARG EXTRAS
RUN printf '%s\n' "${UV_REQUIREMENT}" > /tmp/uv-req.txt \
 && /usr/local/bin/python3.12 -m pip install -q --disable-pip-version-check --require-hashes -r /tmp/uv-req.txt
WORKDIR /env
COPY pyproject.toml uv.lock /env/
RUN uv sync --frozen --no-install-project --python /usr/local/bin/python3.12 --extra train ${EXTRAS} --group hfjobs
ENV VIRTUAL_ENV=/env/.venv PATH=/env/.venv/bin:$PATH
