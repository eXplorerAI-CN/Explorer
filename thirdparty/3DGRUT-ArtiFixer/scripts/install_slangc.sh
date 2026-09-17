#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# Install slangc from official releases into a target directory.
# On ARM64, slangtorch ships x86 binaries, so we need the standalone slangc.

SLANGC_VERSION="2026.5.2"
INSTALL_DIR="${1:-${UV_PROJECT_ENVIRONMENT:-/usr/local}}"

error_exit() {
    echo -e "\033[31m${*}\033[0m" >&2
    exit 1
}

if [ -f "$INSTALL_DIR/bin/slangc" ]; then
    actual_version=$("$INSTALL_DIR/bin/slangc" -version 2>&1 || true)
    if [ "$actual_version" = "$SLANGC_VERSION" ]; then
        echo "slangc $SLANGC_VERSION already installed at $INSTALL_DIR/bin/slangc"
        exit 0
    fi
    echo "slangc version mismatch: expected $SLANGC_VERSION, got $actual_version"
fi

case "$(uname -m)" in
    x86_64|amd64)
        DOWNLOAD_URL="https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-x86_64.tar.gz"
        ;;
    aarch64|arm64)
        DOWNLOAD_URL="https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-aarch64.tar.gz"
        ;;
    *)
        error_exit "Unsupported platform: $(uname -m)"
        ;;
esac

TARBALL="/tmp/$(basename "$DOWNLOAD_URL")"
if [ ! -f "$TARBALL" ]; then
    echo "Downloading slangc $SLANGC_VERSION for $(uname -m)..."
    wget -O "$TARBALL" "$DOWNLOAD_URL"
fi

mkdir -p "$INSTALL_DIR/bin"
tar -xzf "$TARBALL" -C "$INSTALL_DIR"
echo "slangc installed at $INSTALL_DIR/bin/slangc"
