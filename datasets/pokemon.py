"""
Pokemon dataset loader for Tip-Adapter
Loads from HuggingFace datasets:
- Train: andyqmongo/IVL_CLS_pokemon_1_shot
- Test: andyqmongo/pokemon_eval_standard
"""
import os
import os.path as osp
import sys
from PIL import Image
from io import BytesIO

from .utils import Datum, DatasetBase

template = ['a pokemon of {} type.', 'a {} type pokemon.', 'a photo of a {} type pokemon.']


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
                f"If private, please set HF_TOKEN environment variable:\n"
                f"  export HF_TOKEN='your_huggingface_token'\n"
                f"Or login with: huggingface-cli login\n\n"
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


class Pokemon(DatasetBase):
    """Pokemon classification dataset from HuggingFace."""
    
    dataset_dir = 'pokemon'
    
    def __init__(self, root, num_shots):
        self.num_shots = num_shots
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.template = template
        
        train, val, test = self.load_pokemon_datasets()
        
        if num_shots > 0:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def load_pokemon_datasets(self):
        """Load Pokemon datasets from HuggingFace."""
        num_shots = getattr(self, "num_shots", 1)
        if num_shots <= 1:
            train_name = "andyqmongo/IVL_CLS_pokemon_1_shot"
        else:
            train_name = "andyqmongo/IVL_CLS_pokemon_8_shot"
        print(f"Loading Pokemon datasets from HuggingFace (train={train_name}, shots={num_shots})...")
        
        train_dataset = _load_hf_dataset(train_name, "train")
        test_dataset = _load_hf_dataset("andyqmongo/pokemon_eval_standard", "test")
        
        label_to_classname = {}
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
                label_to_classname[label] = classname
            else:
                label = classname_to_label[classname]
            
            img_path = self._save_image(item, f"train_{idx}", classname)
            
            train_data.append(Datum(
                impath=img_path,
                label=label,
                classname=classname
            ))
        
        print(f"\nLoaded {len(train_data)} training samples, {len(classname_to_label)} classes")
        print(f"[DEBUG] Train classes: {sorted(train_classnames)[:10]}{'...' if len(train_classnames) > 10 else ''}")
        
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
        print(f"[DEBUG] Test classes: {sorted(test_classnames)[:10]}{'...' if len(test_classnames) > 10 else ''}")
        
        common_classes = train_classnames & test_classnames
        print(f"[DEBUG] Common classes: {len(common_classes)}")
        if skipped_classes:
            print(f"[DEBUG] Skipped {len(skipped_classes)} test classes not in train: {sorted(skipped_classes)[:5]}...")
        
        if len(test_data) == 0:
            print("[WARNING] No matching classes between train and test!")
            print("[WARNING] Using all test data with new labels...")
            test_data = []
            for idx, item in enumerate(test_dataset):
                classname = self._get_classname(item)
                
                if classname not in classname_to_label:
                    label = len(classname_to_label)
                    classname_to_label[classname] = label
                    label_to_classname[label] = classname
                else:
                    label = classname_to_label[classname]
                
                img_path = self._save_image(item, f"test_{idx}", classname)
                
                test_data.append(Datum(
                    impath=img_path,
                    label=label,
                    classname=classname
                ))
        
        val_size = min(len(test_data) // 5, 100)
        val_data = test_data[:val_size]
        
        return train_data, val_data, test_data
    
    def _get_classname(self, item):
        """Extract classname from dataset item - use primary_type for Pokemon classification."""
        if 'primary_type' in item and item['primary_type']:
            return str(item['primary_type']).lower().replace('_', ' ')
        elif 'label' in item:
            return str(item['label']).lower().replace('_', ' ')
        elif 'class' in item:
            return str(item['class']).lower().replace('_', ' ')
        elif 'category' in item:
            return str(item['category']).lower().replace('_', ' ')
        elif 'name' in item:
            return str(item['name']).lower().replace('_', ' ')
        else:
            for key in item.keys():
                if key != 'image' and isinstance(item[key], str):
                    return str(item[key]).lower().replace('_', ' ')
            return 'unknown'
    
    def _save_image(self, item, prefix, classname):
        """Save image to disk and return path."""
        class_dir = os.path.join(self.image_dir, classname.replace(' ', '_'))
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
                except:
                    pass
        
        return img_path
