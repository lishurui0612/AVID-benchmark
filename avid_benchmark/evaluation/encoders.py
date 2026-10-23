"""Local-only reusable feature/VQA backends. Scoring functions never load them."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np


CLIP_L14_SHA256='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


class EncoderPool:
    def __init__(self,config):
        self.config=config
        self.device=config.get('device','cuda')
        self.batch_size=int(config.get('batch_size',8))
        self.clip_batch_size=int(config.get('clip_batch_size',160))
        if self.clip_batch_size<1:
            raise ValueError('evaluation.clip_batch_size must be positive.')
        self.models={}
        self.protocols={}
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'

    def path(self,name):
        path=Path(self.config.get(name,'')).expanduser()
        if not self.config.get(name) or not path.exists():
            raise FileNotFoundError(f'Configure evaluation.{name} to existing local weights: {path}')
        return str(path)

    def hf(self,name):
        if name=='clip':
            return self._clip()
        import torch
        from transformers import AutoModel,AutoProcessor,SiglipModel
        if name not in self.models:
            path=self.path(name)
            cls={'siglip':SiglipModel}.get(name,AutoModel)
            self.models[name]=(cls.from_pretrained(path,local_files_only=True).to(self.device).eval(),AutoProcessor.from_pretrained(path,local_files_only=True))
        return self.models[name]

    def _clip(self):
        if 'clip' not in self.models:
            path=Path(self.path('clip'))
            if not path.is_file():
                raise ValueError('evaluation.clip must be the original OpenAI ViT-L-14.pt file, not a Hugging Face directory.')
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):
                    digest.update(block)
            if digest.hexdigest()!=CLIP_L14_SHA256:
                raise ValueError('evaluation.clip does not match the paper\'s OpenAI ViT-L/14 checkpoint (SHA-256 mismatch).')
            import clip
            # clip.load preserves native CUDA mixed precision and uses FP32 on CPU.
            model,_=clip.load(str(path),device=self.device,jit=False)
            self.models['clip']=(model.eval(),clip.tokenize)
            self.protocols['clip']={
                'implementation':'openai_clip_native','model':'ViT-L/14',
                'checkpoint_sha256':digest.hexdigest(),'dtype':str(model.dtype),
                'device':str(self.device),'batch_size':self.clip_batch_size,
                'resize':224,'antialias':False,'text_context_length':77,
                'text_image_preprocessing':'raw_tensor_bilinear224_clip_normalize',
            }
        return self.models['clip']

    def _clip_images(self,images):
        import torch
        import torch.nn.functional as F
        model,_=self._clip()
        features=[]
        with torch.inference_mode():
            mean=torch.tensor([.48145466,.4578275,.40821073],device=self.device)[None,:,None,None]
            std=torch.tensor([.26862954,.26130258,.27577711],device=self.device)[None,:,None,None]
            for start in range(0,len(images),self.clip_batch_size):
                pixels=[]
                for im in images[start:start+self.clip_batch_size]:
                    x=torch.from_numpy(np.array(im,dtype=np.float32)).permute(2,0,1)[None].to(self.device)/255
                    pixels.append(F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=False))
                features.append(model.encode_image((torch.cat(pixels)-mean)/std).float().cpu().numpy())
        return np.concatenate(features)

    def images(self,name,images,image_metric=False):
        if name=='clip':
            # Both image identification and text alignment use this exact path.
            return self._clip_images(images)
        import torch
        model,processor=self.hf(name)
        features=[]
        with torch.inference_mode():
            for start in range(0,len(images),self.batch_size):
                chunk=images[start:start+self.batch_size]
                pixels=processor(images=chunk,return_tensors='pt')['pixel_values'].to(self.device)
                features.append(model.get_image_features(pixel_values=pixels).float().cpu().numpy())
        return np.concatenate(features)

    def texts(self,name,texts):
        import torch
        if name=='clip':
            model,tokenize=self._clip()
            features=[]
            with torch.inference_mode():
                for start in range(0,len(texts),self.clip_batch_size):
                    tokens=tokenize(texts[start:start+self.clip_batch_size],context_length=77,truncate=True).to(self.device)
                    features.append(model.encode_text(tokens).float().cpu().numpy())
            return np.concatenate(features)
        model,processor=self.hf(name)
        features=[]
        # SigLIP was trained with fixed padding and pools the last text position.
        text_options={'padding':'max_length','max_length':64} if name=='siglip' else {'padding':True}
        with torch.inference_mode():
            for start in range(0,len(texts),self.batch_size):
                inputs=processor(text=texts[start:start+self.batch_size],truncation=True,return_tensors='pt',**text_options)
                inputs={key:value.to(self.device) for key,value in inputs.items()}
                features.append(model.get_text_features(**inputs).float().cpu().numpy())
        return np.concatenate(features)

    def inception(self,images):
        import torch
        import torch.nn.functional as F
        from torchvision.models import inception_v3
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.transforms.functional import pil_to_tensor
        if 'inception' not in self.models:
            model=inception_v3(weights=None,init_weights=False)
            model.load_state_dict(torch.load(self.path('inception'),map_location='cpu',weights_only=False))
            self.models['inception']=create_feature_extractor(model,return_nodes={'avgpool':'features'}).to(self.device).eval()
        features=[]
        with torch.inference_mode():
            for start in range(0,len(images),self.batch_size):
                tensors=[F.interpolate(pil_to_tensor(im).float().unsqueeze(0)/255,size=(342,342),mode='bilinear',align_corners=False) for im in images[start:start+self.batch_size]]
                pixels=torch.cat(tensors).to(self.device)
                mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
                std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
                features.append(self.models['inception']((pixels-mean)/std)['features'].flatten(1).cpu().numpy())
        return np.concatenate(features)

    def vision(self,name,images):
        """Historical AlexNet/EfficientNet/SwAV nodes; strictly local weights."""
        import torch
        import torch.nn.functional as F
        from torchvision.models import alexnet, efficientnet_b1, resnet50
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.transforms.functional import pil_to_tensor
        if name not in self.models:
            constructors={'alexnet':alexnet,'efficientnet':efficientnet_b1,'swav':resnet50}
            model=constructors[name](weights=None)
            state=torch.load(self.path(name),map_location='cpu',weights_only=False)
            if name=='swav':
                state=state.get('state_dict',state)
                state={k.removeprefix('module.'):v for k,v in state.items()}
                state={k:v for k,v in state.items() if not k.startswith(('projection_head.','prototypes.'))}
                missing,unexpected=model.load_state_dict(state,strict=False)
                if set(missing)-{'fc.weight','fc.bias'} or unexpected:
                    raise ValueError(f'Incompatible SwAV backbone: missing={missing}, unexpected={unexpected}')
            else:
                model.load_state_dict(state,strict=True)
            nodes={'features.4':'alex2','features.11':'alex5'} if name=='alexnet' else {'avgpool':'features'}
            self.models[name]=create_feature_extractor(model,return_nodes=nodes).to(self.device).eval().requires_grad_(False)
        size={'alexnet':256,'efficientnet':255,'swav':224}[name]
        features={}
        with torch.inference_mode():
            for start in range(0,len(images),self.batch_size):
                tensors=[F.interpolate(pil_to_tensor(im).float().unsqueeze(0)/255,size=(size,size),mode='bilinear',align_corners=False) for im in images[start:start+self.batch_size]]
                pixels=torch.cat(tensors).to(self.device)
                mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
                std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
                for node,value in self.models[name]((pixels-mean)/std).items():
                    features.setdefault(node,[]).append(value.flatten(1).float().cpu().numpy())
        return {node:np.concatenate(values) for node,values in features.items()}

    def tifa(self,images,captions):
        import sys
        vendor=Path(__file__).resolve().parents[1]/'vendor/tifa_official_8e37050'
        sys.path.insert(0,str(vendor))
        from ._backends.simple_attribute_metrics import TifaQuestionBackend
        if 'tifa' not in self.models:
            self.models['tifa']=TifaQuestionBackend(self.path('tifa_mplug'),self.path('tifa_sbert'))
        backend=self.models['tifa']
        bank=json.loads(Path(self.path('tifa_bank')).read_text(encoding='utf-8'))
        lookup={e['caption']:e for e in bank['entries']}
        scores=[]
        for image,caption in zip(images,captions):
            if caption not in lookup:
                raise ValueError(f'TIFA bank is missing reference caption: {caption}')
            qas=lookup[caption]['qas']
            question_scores=[]
            for qa in qas:
                _,answer=backend.answer(image,qa['question'],qa['choices'])
                question_scores.append(float(answer==qa['answer']))
            scores.append(question_scores)
        return scores
