import subprocess

p1 = subprocess.run(["iree-benchmark-module", "--list_devices"], capture_output=True, text=True)
p2 = subprocess.run(["rocm-smi", "--showuniqueid"], capture_output=True, text=True)
p3 = subprocess.run(["rocminfo"], capture_output=True, text=True)

id = 0
iree = []
for line in p1.stdout.split('\n'):
    if line.startswith('hip'):
        bogus_uid = line[10:]
        uid = bogus_uid.replace('-', '')
        hex_uid = ''.join(chr(int(uid[i:i+2], base=16)) for i in range(0, len(uid), 2))
        iree.append((id, bogus_uid, hex_uid))
        id += 1

for line in p2.stdout.split('\n'):
    if line.startswith('GPU'):
        gpu_id = int(line[4:line.index(']')])
        uid = line[line.index('x')+1:]
        uid = '0' * (16-len(uid)) + uid

        for i, t in enumerate(iree):
            if t[2] == uid:
                iree[i] += (gpu_id,)

uid = None
for line in p3.stdout.split('\n'):
    if 'Uuid:' in line and 'GPU-' in line:
        line = line.strip()
        uid = line[line.index('GPU-')+4:]
    if uid is not None and 'Node:' in line:
        line = line.strip()
        gpu_id = int(line[line.rindex(' ')+1:])
        for i, t in enumerate(iree):
            if t[2] == uid:
                iree[i] += (gpu_id,)

print('IREE id, IREE UUID, ROCm UUID, ROCM id, HSA Node')
for row in iree:
    print(*row)
