# ==========================================================
# Stage 1: Build Nativo do llama.cpp em C++ (Otimizado)
# ==========================================================
FROM docker.io/library/alpine:3.20 AS builder-native

RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    linux-headers

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

WORKDIR /src/llama.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON
RUN cmake --build build --config Release -j$(nproc) --target llama-cli llama-quantize llama-imatrix

# ==========================================================
# Stage 2: Runtime Environment (Rootless Podman Friendly)
# ==========================================================
FROM docker.io/library/python:3.11-slim

LABEL maintainer="Maurício Helfstein Gonçalves"
LABEL project="MAURICE"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/workspace/bin:${PATH}"

# Instalação de dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Criação de usuário não-root para compatibilidade rootless com Podman
RUN useradd -m -u 1001 -s /bin/bash maurice
USER maurice
WORKDIR /home/maurice/app

# Cópia dos binários C++ construídos no Stage 1
COPY --from=builder-native /src/llama.cpp/build/bin/llama-cli /home/maurice/bin/llama-cli
COPY --from=builder-native /src/llama.cpp/build/bin/llama-quantize /home/maurice/bin/llama-quantize
COPY --from=builder-native /src/llama.cpp/build/bin/llama-imatrix /home/maurice/bin/llama-imatrix

# Configuração do ambiente Python
COPY --chown=maurice:maurice pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --user -r requirements.txt || pip install --no-cache-dir --user torch transformers datasets trl peft unsloth

# Cópia do código-fonte do projeto
COPY --chown=maurice:maurice . .

ENV PATH="/home/maurice/.local/bin:/home/maurice/bin:${PATH}"

ENTRYPOINT ["/bin/bash"]
