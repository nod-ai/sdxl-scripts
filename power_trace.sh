#!/bin/bash

set -xe

readonly tmpdir="$(mktemp -d -t "traceXXXXXX")"

cleanup () {
    rm -r "$tmpdir"
}
trap cleanup EXIT

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly real_iree="$(command -v iree-benchmark-module)"

if [ -z "$real_iree" ]; then
    echo "iree-benchmark-module not found"
    exit 1
fi

cat << EOF > "$tmpdir/doas-root"
set -xe
rm -rf /var/tmp/smutraceTmpFiles
mkdir -p /var/tmp/smutraceTmpFiles
ulimit -n 100000
LD_LIBRARY_PATH=/usr/local/lib runTracer.sh "$real_iree" "\$@"
EOF

cat << EOF > "$tmpdir/iree-benchmark-module"
#!/bin/bash
set -xe
sudo bash "$tmpdir/doas-root" "\$@"
ret=$?
python3 "$script_dir/corellator.py" trace.rpd || echo "Returned $?"
exit $ret
EOF

chmod a+x "$tmpdir/iree-benchmark-module"

export PATH="$tmpdir:$PATH"

"$@"
