import json

v9 = {m['name']: m for m in json.load(open('weights_final_v6_testcalib/calib_v9.json'))['members']}
issues = []
for name, m in v9.items():
    own = json.load(open(f'{name}/calib.json'))
    fields = {'backbone': own['backbone'], 'size': int(own['size']),
              'preproc': own.get('preproc'), 'proj': own.get('proj', 'mip'),
              'stack': own.get('stack')}
    for f in ['backbone', 'size', 'preproc', 'proj', 'stack']:
        if m[f] != fields[f]:
            issues.append(f'{name}.{f}: v9={m[f]} own={fields[f]}')
    print(f"{name}: own preproc={fields['preproc']} proj={fields['proj']} "
          f"stack={fields['stack']} size={fields['size']} bb={fields['backbone']}")
print()
if issues:
    print('!!! MISMATCHES:')
    [print(' ', i) for i in issues]
else:
    print('ALL MEMBER CONFIGS MATCH their calib.json')