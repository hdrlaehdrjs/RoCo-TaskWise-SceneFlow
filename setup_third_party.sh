#!/bin/bash
# Fetch DEFOM-Stereo and flow_library into third_party/ at the commits used in the paper,
# and patch DEFOM-Stereo.
set -e
cd "$(dirname "$0")"
DEFOM_COMMIT=5b27591d83956e5d329ff66704271a9e68b1c75f
FLOW_LIBRARY_COMMIT=8454aed75172b230304ea9942b95626b99106534

mkdir -p third_party
if [ ! -d third_party/DEFOM-Stereo ]; then
    git clone https://github.com/Insta360-Research-Team/DEFOM-Stereo.git third_party/DEFOM-Stereo
    git -C third_party/DEFOM-Stereo checkout --quiet "$DEFOM_COMMIT"
    git -C third_party/DEFOM-Stereo apply ../../patches/defom_stereo_detach.patch
fi
if [ ! -d third_party/flow_library ]; then
    git clone https://github.com/cv-stuttgart/flow_library.git third_party/flow_library
    git -C third_party/flow_library checkout --quiet "$FLOW_LIBRARY_COMMIT"
fi
