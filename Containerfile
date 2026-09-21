# ==========================================================
# Stage 1: Build Nativo do llama.cpp com Arquitetura Genérica
# ==========================================================
FROM docker.io/library/debian:bookworm-slim AS builder-native

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

WORKDIR /src/llama.cpp

# Desativa expressamente instruções nativas específicas da máquina hospedeira
# e compila para x86-64 genérico sem dependência de extensões restritas
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DCMAKE_C_FLAGS="-march=x86-64 -mtune=generic" \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mtune=generic" \
    -DCMAKE_INSTALL_PREFIX=/install

RUN cmake --build build --config Release -j$(nproc) --target llama-cli llama-quantize llama-imatrix
RUN cmake --install build --component default || cp build/bin/llama-* /install/bin/ || true

# ==========================================================
# Stage 2: Runtime Environment (Debian Bookworm)
# ==========================================================
FROM docker.io/library/python:3.11-slim-bookworm

LABEL maintainer="Maurício Helfstein Gonçalves"
LABEL project="MAURICE"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:/home/maurice/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 -s /bin/bash maurice

# Copia os binários compilados e instalados
COPY --from=builder-native /src/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder-native /src/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=builder-native /src/llama.cpp/build/bin/llama-imatrix /usr/local/bin/llama-imatrix

RUN chmod +x /usr/local/bin/llama-*

USER maurice
WORKDIR /home/maurice/app

COPY --chown=maurice:maurice pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --user -r requirements.txt 2>/dev/null || true

COPY --chown=maurice:maurice . .

ENTRYPOINT ["/bin/bash"]
