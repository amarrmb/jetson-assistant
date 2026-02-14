#!/bin/bash
# Generate ALSA config from ALSA_CARD env var if set.
# This lets users pick their audio device without needing ~/.asoundrc on the host.
#
# Usage: ALSA_CARD=USB docker compose up -d
#   (where "USB" matches part of the card name from "aplay -l")

if [ -n "$ALSA_CARD" ]; then
    cat > /root/.asoundrc << EOF
pcm.!default { type plug slave.pcm "hw:${ALSA_CARD}" }
ctl.!default { type hw card "${ALSA_CARD}" }
EOF
fi

exec jetson-assistant "$@"
