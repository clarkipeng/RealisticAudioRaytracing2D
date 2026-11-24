#!/bin/zsh
# Builds Assets/Plugins/macOS/AcousticRT.bundle (arm64 and x86_64) from AcousticRT.mm with Unity's plugin headers.
#   Native/AcousticRT/build.sh [/path/to/Unity.app]
set -e
here=${0:A:h}
root=${here:h:h}
unity=${1:-/Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app}
bundle=$root/Assets/Plugins/macOS/AcousticRT.bundle
mkdir -p $bundle/Contents/MacOS
clang++ -std=c++17 -O2 -fobjc-arc -arch arm64 -arch x86_64 -mmacosx-version-min=13.0 -bundle \
  -I $unity/Contents/Resources/PluginAPI -framework Metal -framework Foundation \
  $here/AcousticRT.mm -o $bundle/Contents/MacOS/AcousticRT
cat > $bundle/Contents/Info.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>AcousticRT</string>
<key>CFBundleIdentifier</key><string>acoustics.AcousticRT</string>
<key>CFBundlePackageType</key><string>BNDL</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>AcousticRT</string>
<key>CFBundleVersion</key><string>1</string>
</dict></plist>
PLIST
codesign --force --sign - $bundle >/dev/null 2>&1 || true
echo built $bundle
