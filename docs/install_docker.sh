#!/bin/bash
# Time-stamp: "2024-11-07 20:00:30 (ywatanabe)"
# File: ./mngs_repo/docs/install_docker.sh


#!/bin/bash
# File: ./mngs_repo/docs/install_docker.sh

readonly DOCKER_GROUP="docker"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    sudo apt install docker.io -y || {
        log "Error: Docker installation failed"
        exit 1
    }
    sudo usermod -aG $DOCKER_GROUP "$USER" || {
        log "Error: Failed to add user to docker group"
        exit 1
    }
    log "Docker installed. Please run 'newgrp docker' or log out and back in."
else
    log "Docker is already installed"
fi


# EOF
