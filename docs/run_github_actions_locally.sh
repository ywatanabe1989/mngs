#!/bin/bash
# Time-stamp: "2024-11-07 20:21:31 (ywatanabe)"
# File: ./mngs_repo/docs/run_github_actions_locally.sh


readonly LOG_DIR="$(dirname "$0")"
readonly LOG_PATH="$LOG_DIR/$(basename "$0")-$(date +"%Y-%m-%d_%H-%M-%S").log"

setup_env() {
    cat > .env.act << EOL
GITHUB_TOKEN=${GITHUB_TOKEN:-}
PYTHON_VERSION=3.8
EOL
}

test_workflow() {
    local workflow_file="$1"
    local log_file="$LOG_DIR/$(basename "$workflow_file").log"
    (act -W "$workflow_file" --env-file .env.act > "$log_file" 2>&1) &
}

test_workflows() {
    local workflow_dir=".github/workflows"
    
    if [ ! -d "$workflow_dir" ]; then
        echo "Error: Workflows directory not found: $workflow_dir"
        exit 1
    fi

    if ! command -v act &> /dev/null; then
        curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
    fi

    setup_env

    echo "Running all workflows in parallel..."
    
    for workflow in "$workflow_dir"/*.{yml,yaml}; do
        if [ -f "$workflow" ]; then
            echo "Starting workflow: $workflow"
            test_workflow "$workflow"
        fi
    done

    # Wait for all background jobs
    wait
}

mkdir -p "$LOG_DIR"
test_workflows 2>&1 | tee "$LOG_PATH"
echo -e "\nLogged to: $LOG_PATH"

# EOF

# EOF
# #!/bin/bash
# # Time-stamp: "2024-11-07 20:13:42 (ywatanabe)"
# # File: ./mngs_repo/docs/run_github_actions_locally.sh


# readonly LOG_DIR="$(dirname "$0")"
# readonly LOG_PATH="$LOG_DIR/$(basename "$0")-$(date +"%Y-%m-%d_%H-%M-%S").log"

# setup_env() {
#     cat > .env.act << EOL
# GITHUB_TOKEN=${GITHUB_TOKEN:-}
# PYTHON_VERSION=3.8
# EOL
# }

# test_workflow() {
#     local workflow_file="$1"
#     act -W "$workflow_file" --env-file .env.act
#     return $?
# }

# test_workflows() {
#     local workflow_dir=".github/workflows"
    
#     if [ ! -d "$workflow_dir" ]; then
#         echo "Error: Workflows directory not found: $workflow_dir"
#         exit 1
#     fi

#     if ! command -v act &> /dev/null; then
#         curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
#     fi

#     setup_env

#     echo "Running all workflows..."
#     local exit_status=0
    
#     for workflow in "$workflow_dir"/*.{yml,yaml}; do
#         if [ -f "$workflow" ]; then
#             echo "Running workflow: $workflow"
#             if ! test_workflow "$workflow"; then
#                 echo "Error: Workflow $workflow failed"
#                 exit_status=1
#             fi
#         fi
#     done

#     return $exit_status
# }

# mkdir -p "$LOG_DIR"
# test_workflows 2>&1 | tee "$LOG_PATH"
# echo -e "\nLogged to: $LOG_PATH"

# 

# EOF
