with open('/mnt/c/think-engine/data/mbe_master_ja.md', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('1000ユーザーが同じクラス: 000収益 vs $0.001コスト', '1000ユーザーが同じクラス: $1000収益 vs $0.001コスト')
with open('/mnt/c/think-engine/data/mbe_master_ja.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed $1000')
