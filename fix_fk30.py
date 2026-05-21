with open('/mnt/c/think-engine/data/mbe_master_ja.md', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('/usr/bin/bash.001', '$0.001')
content = content.replace('/usr/bin/bash.01', '$0.01')
content = content.replace('000収益 vs /usr/bin/bash.001コスト', '$1000収益 vs $0.001コスト')
with open('/mnt/c/think-engine/data/mbe_master_ja.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed OK')
