#!/usr/bin/env bash
set -euo pipefail

PORT="40000"
IMAGE="jekyll-site-builder"

docker build -t "$IMAGE" .
echo "Serving site at http://localhost:$PORT/ (Ctrl-C to stop)"
docker run --rm -it \
  --name=larryfenn \
  --network=medianet \
  -p "$PORT:4000" \
  -v "$(pwd):/site" \
  -v /site/node_modules \
  "$IMAGE" \
  bundle exec jekyll serve --host 0.0.0.0 --port 4000 --livereload --watch --drafts
