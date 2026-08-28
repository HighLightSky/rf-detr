FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY environment.yml /tmp/environment.yml
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu121
RUN micromamba create -y -n detector -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/detector/bin:/opt/micromamba/bin:$PATH

COPY app /app
COPY models /app/models

RUN mkdir -p /input /output

ENTRYPOINT ["python", "/app/main.py"]
