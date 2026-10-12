import sys
sys.path.insert(0, '.')
from trustagent.retrieval.retrieve import retrieve

results = retrieve('Fatshark Teleporter headset', k=3)
print('--- Checking image_path in Qdrant records ---')
for r in results:
    entity = r['record'].get('entity', 'N/A')[:55]
    img    = r['record'].get('image_path', '')
    label  = img[:90] if img else '(empty - no image in dataset record)'
    print(f'  Product: {entity}')
    print(f'  Image  : {label}')
    print()

# Also test AirPods (the product from your screenshot)
results2 = retrieve('AirPods Pro 2', k=1)
print('--- AirPods Pro 2 ---')
for r in results2:
    entity = r['record'].get('entity', 'N/A')[:55]
    img    = r['record'].get('image_path', '')
    label  = img[:90] if img else '(empty)'
    print(f'  Product: {entity}')
    print(f'  Image  : {label}')
