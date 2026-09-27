#!/bin/zsh

# Keep paths reliable even when this file is opened from Finder.
SCRIPT_FOLDER="${0:A:h}"
cd "$SCRIPT_FOLDER" || exit 1

exec python3 dolphin_interface.py "$@"
