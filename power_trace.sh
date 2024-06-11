#!/bin/bash

set -xe

tmpdir=$(mktemp -d -t "traceXXXXXX")

cleanup () {
    rm -r $tmpdir
}
trap cleanup EXIT

script_dir=$(dirname $(realpath $0))

real_iree=$(command -v iree-benchmark-module)


cat << EOF > $tmpdir/iree-benchmark-module
#!/bin/bash
sudo LD_LIBRARY_PATH=/usr/local/lib runTracer.sh $real_iree \$@
ret=$?
python3 $script_dir/corellator.py trace.rpd || echo "Returned $?"
exit $ret
EOF

chmod a+x $tmpdir/iree-benchmark-module

export PATH=$tmpdir:$PATH

source $@
