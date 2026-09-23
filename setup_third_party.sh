#!/bin/bash
# Fetch DEFOM-Stereo and flow_library into third_party/ and patch DEFOM-Stereo.
set -e
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/DEFOM-Stereo ]; then
    git clone https://github.com/Insta360-Research-Team/DEFOM-Stereo.git third_party/DEFOM-Stereo
    (cd third_party/DEFOM-Stereo && git apply ../../patches/defom_stereo_detach.patch)
fi
if [ ! -d third_party/flow_library ]; then
    git clone https://github.com/cv-stuttgart/flow_library.git third_party/flow_library
fi
