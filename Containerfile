# ==========================================================
# Stage 1: Build Nativo usando a mesma base glibc
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
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON
RUN cmake --build build --config Release -j$(nproc) --target llama-cli llama-quantize llama-imatrix

# ==========================================================
# Stage 2: Runtime Environment (glibc compativel)
# ==========================================================
FROM docker.io/library/python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 -s /bin/bash maurice
USER maurice
WORKDIR /home/maurice/app

# Copiar binarios compilados com glibc
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-cli /home/maurice/bin/llama-cli
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-quantize /home/maurice/bin/llama-quantize
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/llama-imatrix /home/maurice/bin/llama-imatrix

RUN chmod +x /home/maurice/bin/*

ENV PATH="/home/maurice/.local/bin:/home/maurice/bin:${PATH}"
ENTRYPOINT ["/bin/bash"]
