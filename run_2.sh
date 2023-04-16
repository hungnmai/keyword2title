#!/bin/bash
uvicorn "routers_2:app" --host 0.0.0.0 --port 8098 --workers 1 --timeout-keep-alive 600