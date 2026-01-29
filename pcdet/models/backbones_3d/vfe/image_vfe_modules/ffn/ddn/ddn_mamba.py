from pcdet.models.backbones_3d.vfe.Vmamba.vmamba import MM_VSSM
from collections import OrderedDict
from .ddn_template import DDNTemplate
import torch.nn as nn
try:
    import torchvision
except:
    pass

class DDNMamba(DDNTemplate):

    def __init__(self, backbone_name, **kwargs):
        if backbone_name == "VMamba":
            constructor = MM_VSSM
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
        
        self.lin = nn.ModuleList([nn.Linear(3*(2**(5+n)), 2**(8+n)) for n in range(len(self.feat_extract_layer))])
        
    def get_model(self, constructor):
        # Get model
            model = constructor( 
                    out_indices=(0, 1, 2, 3),
                    pretrained=self.pretrained_path,
                    dims=96,
                    depths=(2, 2, 5, 2),
                    ssm_d_state=1,
                    ssm_dt_rank="auto",
                    ssm_ratio=2.0,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    forward_type="v05_noz",
                    mlp_ratio=4.0,
                    downsample_version="v3",
                    patchembed_version="v2",
                    drop_path_rate=0.2
            )

            # Update weights
            '''if self.pretrained_path is not None:
                model_dict = model.state_dict()

                # Download pretrained model if not available yet
                checkpoint_path = Path(self.pretrained_path)
                if not checkpoint_path.exists():
                    checkpoint = checkpoint_path.name
                    save_dir = checkpoint_path.parent
                    #save_dir.mkdir(parents=True)
                    url = f'https://download.pytorch.org/models/{checkpoint}'
                    hub.load_state_dict_from_url(url, save_dir)

                # Get pretrained state dict
                pretrained_dict = torch.load(self.pretrained_path)
                #pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                #                                              pretrained_dict=pretrained_dict)

                # Update current model state dict
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)'''

            return model
    
    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()

        if isinstance(self.feat_extract_layer, list):
            features = self.model(x)
            for index, feature_name in enumerate(self.feat_extract_layer): # [B, 256, 129, 484] [B, 512, 65, 242] [B, 1024, 65, 242] [B, 2048, 65, 242]
                #print(index,features[index].shape)
                obj = features[index]
                obj = self.lin[index](obj.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                result[feature_name] = obj
            if self.use_lidar_depth or (not self.use_depth):
                pass
            else:
                x = features[f'features_{len(self.feat_extract_layer) - 1}']
                feat_shape = features['features_0'].shape[-2:]
                x = self.model.classifier(x) # [2, 81, 65, 242]
                result["logits_small"] = x
                x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
                result["logits"] = x
        else:
            if self.use_lidar_depth or (not self.use_depth):
                features = self.model_simple(x)
                result['features'] = features # [1, 256, 129, 484]
            else:
                features = self.model.backbone(x)
                result['features'] = features['features'] # [1, 256, 129, 484]
                feat_shape = features['features'].shape[-2:]
                # Prediction classification logits
                x = features["out"] # [1, 2048, 65, 242]
                x = self.model.classifier(x) # [2, 81, 65, 242]
                x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
                result["logits"] = x

        # Prediction auxillary classification logits
        '''if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x'''

        return result