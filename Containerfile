# ==========================================================
# Stage 1: Build Nativo do llama.cpp com glibc (Debian Bookworm)
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
# Desativa bibliotecas compartilhadas para gerar binários autocontidos
# Mantém flags genéricas para rodar em qualquer runner do GitHub Actions
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF
RUN cmake --build build --config Release -j$(nproc) --target llama-cli llama-quantize llama-imatrix

# ==========================================================
# Stage 2: Runtime Environment (Debian Bookworm Python)
# ==========================================================
FROM docker.io/library/python:3.11-slim-bookworm

LABEL maintainer="Maurício Helfstein Gonçalves"
LABEL project="MAURICE"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/maurice/bin:/home/maurice/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 -s /bin/bash maurice
USER maurice
WORKDIR /home/maurice/app

# Cria diretório de binários
RUN mkdir -p /home/maurice/bin

# Copia os binários monolíticos
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-cli /home/maurice/bin/llama-cli
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-quantize /home/maurice/bin/llama-quantize
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-imatrix /home/maurice/bin/llama-imatrix

RUN chmod +x /home/maurice/bin/*

COPY --chown=maurice:maurice pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --user -r requirements.txt 2>/dev/null || true

COPY --chown=maurice:maurice . .

ENTRYPOINT ["/bin/bash"]
