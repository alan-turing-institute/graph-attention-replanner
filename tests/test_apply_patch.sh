#/bin/bash

if [ -d "../graph-attention-replanner/external/rl4co/rl4co/envs/routing/mtsp_custom" ]; then
    echo "Patch was applied successfully."
else
    echo "Missing ../graph-attention-replanner/external/rl4co/rl4co/envs/routing/mtsp_custom directory, patch have not been applied correctly."
fi