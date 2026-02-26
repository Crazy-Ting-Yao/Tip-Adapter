"""
MVTec AD dataset loader for Tip-Adapter
Loads from HuggingFace: andyqmongo/IVL_OOD_MVTec_{object_name}
Objects: bottle, cable, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
- Train split: 1_shot
- Test split: test
"""
import os
import os.path as osp
import sys
from PIL import Image
from io import BytesIO

from .utils import Datum, DatasetBase

MVTEC_OBJECTS = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


def _load_hf_dataset(dataset_name, split="train"):
    """Load dataset from HuggingFace, avoiding namespace conflict."""
    import subprocess
    import json
    
    hf_token = os.environ.get('HF_TOKEN', '')
    token_arg = f'token="{hf_token}"' if hf_token else ''
    
    script = f'''
import sys
import os
sys.path = [p for p in sys.path if 'Tip-Adapter' not in p]
os.environ['HF_TOKEN'] = '{hf_token}'
from datasets import load_dataset
try:
    ds = load_dataset("{dataset_name}", split="{split}", {token_arg})
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
data = []
for item in ds:
    row = {{}}
    for k, v in item.items():
        if k == 'image':
            from io import BytesIO
            import base64
            buf = BytesIO()
            v.save(buf, format='PNG')
            row[k] = base64.b64encode(buf.getvalue()).decode()
        else:
            row[k] = v
    data.append(row)
import json
print(json.dumps(data))
'''
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd='/'
    )
    
    if result.returncode != 0:
        error_msg = result.stderr
        if "doesn't exist" in error_msg or "cannot be accessed" in error_msg:
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found or is private.\n"
                f"If private, please set HF_TOKEN environment variable.\n"
                f"Error: {error_msg}"
            )
        raise RuntimeError(f"Failed to load dataset: {error_msg}")
    
    import base64
    data = json.loads(result.stdout)
    
    for item in data:
        if 'image' in item:
            img_bytes = base64.b64decode(item['image'])
            item['image'] = Image.open(BytesIO(img_bytes))
    
    return data


def _template_for_object(obj_name):
    """Template for MVTec object."""
    return [
        f'a photo of a {{}} {obj_name.replace("_", " ")}.',
        f'a {obj_name.replace("_", " ")} with {{}} defect.',
        f'{obj_name.replace("_", " ")} inspection showing {{}}.'
    ]


class MVTec(DatasetBase):
    """MVTec AD dataset from HuggingFace. object_name: bottle, cable, carpet, etc."""
    
    def __init__(self, root, num_shots, object_name='bottle'):
        self.object_name = object_name
        self.dataset_dir = os.path.join(root, f'mvtec_{object_name}')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.template = _template_for_object(object_name)
        
        train, val, test = self.load_mvtec_datasets()
        
        if num_shots > 0:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def load_mvtec_datasets(self):
        """Load MVTec datasets from HuggingFace."""
        hf_name = f"andyqmongo/IVL_OOD_MVTec_{self.object_name}"
        print(f"Loading MVTec ({self.object_name}) from HuggingFace...")
        
        train_dataset = _load_hf_dataset(hf_name, "1_shot")
        test_dataset = _load_hf_dataset(hf_name, "test")
        
        classname_to_label = {}
        
        print(f"\n[DEBUG] Sample train item keys: {list(train_dataset[0].keys()) if train_dataset else 'empty'}")
        print(f"[DEBUG] Sample test item keys: {list(test_dataset[0].keys()) if test_dataset else 'empty'}")
        
        if train_dataset:
            sample = train_dataset[0]
            sample_info = {k: v for k, v in sample.items() if k != 'image'}
            print(f"[DEBUG] Train sample (non-image): {sample_info}")
        if test_dataset:
            sample = test_dataset[0]
            sample_info = {k: v for k, v in sample.items() if k != 'image'}
            print(f"[DEBUG] Test sample (non-image): {sample_info}")
        
        train_data = []
        train_classnames = set()
        for idx, item in enumerate(train_dataset):
            classname = self._get_classname(item)
            train_classnames.add(classname)
            
            if classname not in classname_to_label:
                label = len(classname_to_label)
                classname_to_label[classname] = label
            else:
                label = classname_to_label[classname]
            
            img_path = self._save_image(item, f"train_{idx}", classname)
            
            train_data.append(Datum(
                impath=img_path,
                label=label,
                classname=classname
            ))
        
        print(f"\nLoaded {len(train_data)} training samples, {len(classname_to_label)} classes")
        print(f"[DEBUG] Train classes: {sorted(train_classnames)}")
        
        test_data = []
        test_classnames = set()
        skipped_classes = set()
        for idx, item in enumerate(test_dataset):
            classname = self._get_classname(item)
            test_classnames.add(classname)
            
            if classname not in classname_to_label:
                skipped_classes.add(classname)
                continue
            
            label = classname_to_label[classname]
            img_path = self._save_image(item, f"test_{idx}", classname)
            
            test_data.append(Datum(
                impath=img_path,
                label=label,
                classname=classname
            ))
        
        print(f"Loaded {len(test_data)} test samples (from {len(test_dataset)} total)")
        print(f"[DEBUG] Test classes: {sorted(test_classnames)}")
        print(f"[DEBUG] Common classes: {len(train_classnames & test_classnames)}")
        if skipped_classes:
            print(f"[DEBUG] Skipped test classes not in train: {sorted(skipped_classes)}")
        
        if len(test_data) == 0:
            print("[WARNING] No matching classes; using all test data with new labels...")
            test_data = []
            for idx, item in enumerate(test_dataset):
                classname = self._get_classname(item)
                if classname not in classname_to_label:
                    label = len(classname_to_label)
                    classname_to_label[classname] = label
                else:
                    label = classname_to_label[classname]
                img_path = self._save_image(item, f"test_{idx}", classname)
                test_data.append(Datum(impath=img_path, label=label, classname=classname))
        
        val_size = min(len(test_data) // 5, 100)
        val_data = test_data[:val_size]
        
        return train_data, val_data, test_data
    
    def _get_classname(self, item):
        """Extract classname from dataset item."""
        if 'label' in item and item['label'] is not None:
            return str(item['label']).lower().replace('_', ' ')
        if 'defect' in item and item['defect'] is not None:
            return str(item['defect']).lower().replace('_', ' ')
        if 'class' in item and item['class'] is not None:
            return str(item['class']).lower().replace('_', ' ')
        if 'anomaly' in item and item['anomaly'] is not None:
            return str(item['anomaly']).lower().replace('_', ' ')
        if 'category' in item and item['category'] is not None:
            return str(item['category']).lower().replace('_', ' ')
        for key in item.keys():
            if key != 'image' and item[key] is not None and isinstance(item[key], str):
                return str(item[key]).lower().replace('_', ' ')
        return 'unknown'
    
    def _save_image(self, item, prefix, classname):
        """Save image to disk and return path."""
        class_dir = os.path.join(self.image_dir, classname.replace(' ', '_').replace('/', '_'))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{prefix}.jpg")
        if os.path.exists(img_path):
            return img_path
        if 'image' in item:
            img = item['image']
            if isinstance(img, Image.Image):
                img.convert('RGB').save(img_path)
            elif isinstance(img, bytes):
                Image.open(BytesIO(img)).convert('RGB').save(img_path)
            elif isinstance(img, dict) and 'bytes' in img:
                Image.open(BytesIO(img['bytes'])).convert('RGB').save(img_path)
            else:
                try:
                    img.save(img_path)
                except Exception:
                    pass
        return img_path
