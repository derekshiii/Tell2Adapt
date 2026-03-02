import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import os
import numpy as np
import time
from safetensors.torch import load_file
from Prompt_regulization import correct_text
def process_multi_prompts(text):
    """
    Process the input text to handle multiple prompts.
    This function splits the text by [SEP] and returns a list of prompts.
    """
    if text is None:
        return None, None
    # ensure we have a list of strings
    text = text if isinstance(text, (list, tuple)) else [text]
    text = [_text.split("[SEP]")for _text in text]    # split text by [SEP]
    num_prompts = torch.tensor([len(_text) for _text in text], dtype=torch.int64)
    # flatten multiple text prompts to the batch dimension
    # intial format: [(text1 for img1, text2 for img1), (text1 for img2)]
    text = [t for i in range(len(text)) for t in text[i]]
    # new format: [text1 for img1, text2 for img1, text1 for img2]
    return text, num_prompts
        
        
def tile_feature(feat: torch.Tensor, P: int) -> torch.Tensor:
    # feat: [B, C, H, W], P = num_prompts
    B, C, H, W = feat.shape
    # 1) insert prompt dim
    v = feat.view(B, 1, C, H, W)           # [B, 1, C, H, W]
    # 2) virtually expand along prompt dim
    e = v.expand(-1, P, -1, -1, -1)        # [B, P, C, H, W]
    # 3) collapse back to batch
    return e.reshape(B * P, C, H, W)       # [B*P, C, H, W]

class MaskFormerHead(nn.Module):
    def __init__(self, pixel_decoder, predictor):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        self.classes = [
            "liver",
            "lung",
            "kidney",
            "pancreas",
            "heart anatomies",
            "brain anatomies",
            "eye anatomies",
            "vessel",
            "other organ",
            "tumor",
            "infection",
            "other lesion",
            "fluid disturbance",
            "other abnormality",
            "histology structure",
            "other",
            "background",
        ]

    def encode_prompts(self, text, eval=True):
        self.predictor.language_encoder.get_text_embeddings(self.classes, is_eval=eval)
        
        extra = {}
        print(f'Text Prompts:{text}')#Process 
        text = correct_text(text)
        print(f'Regulized Text Prompts: {text}')
        logit_scale = self.predictor.language_encoder.logit_scale
        text, num_prompts = process_multi_prompts(text)
        gtext = self.predictor.language_encoder.get_text_token_embeddings(
            text, name="grounding", token=False, norm=False
        )
        token_emb = gtext["token_emb"]
        tokens = gtext["tokens"]
        class_emb = gtext["class_emb"]
        query_emb = nn.utils.rnn.pad_sequence(
            [
                _token_emb[_tokens.bool()]
                for _token_emb, _tokens in zip(
                    token_emb, tokens["attention_mask"]
                )
            ],
            padding_value=-1,
        )

        non_zero_query_mask = query_emb.sum(dim=-1) == -query_emb.shape[-1]
        query_emb[non_zero_query_mask] = 0
            
        extra["grounding_tokens"] = query_emb    # [seq_len, batch_size, dim]
        # extra["grounding_nonzero_mask"] = non_zero_query_mask.t()    # [batch_size, seq_len]
        
        extra["class_emb"] = class_emb    # [batch_size, dim]
        extra["logit_scale"] = logit_scale
        extra["num_prompts"] = num_prompts
        extra["text"] = text
        
        return extra
        

    def forward(self, image_features, prompt_features):
        
        mask_features, _, multi_scale_features = (
            self.pixel_decoder.forward_features(image_features)
        )
        
        num_prompts = prompt_features["num_prompts"]
        
        # repeat image features for each text prompt
        P = int(num_prompts[0])
        if max(num_prompts) > min(num_prompts):
            num_prompts = num_prompts.to(mask_features.device)
            # repeat interleave image features for each text prompt
            mask_features = mask_features.repeat_interleave(
                num_prompts, dim=0
            )
            multi_scale_features = [
                _feature.repeat_interleave(num_prompts, dim=0)
                for _feature in multi_scale_features
            ]
        else:
            # assume same number of prompts for all images
            mask_features = tile_feature(mask_features, P)
            multi_scale_features = [
                tile_feature(_feature, P) for _feature in multi_scale_features
            ]
            
        predictions = self.predictor(
            x=multi_scale_features, mask_features=mask_features, extra=prompt_features
        )
        
        predictions["class_emb"] = prompt_features["class_emb"]
        predictions["logit_scale"] = prompt_features["logit_scale"]

        return predictions


    def override_input_shape(self, input_shape):
        self.pixel_decoder.override_input_shape(input_shape)


class BiomedParseModel(nn.Module):
    def __init__(
        self,
        backbone,
        sem_seg_head,
        pixel_mean=[123.675, 116.280, 103.530],
        pixel_std=[58.395, 57.120, 57.375],
        gray_scale=True,
        convolute_outputs=True,  # for upscaling the output of the model
        out_channels_1=10,  # Parameter for upscaling
        edge_queries=0,    # Number of queries used for edge detection
        enable_visualization=False,  # 添加可视化控制参数
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.sem_seg_head.override_input_shape(backbone.output_shape())
        if gray_scale:
            # same value for all channels
            mean = sum(pixel_mean) / len(pixel_mean)
            std = sum(pixel_std) / len(pixel_std)
            pixel_mean = [mean for _ in range(3)]
            pixel_std = [std for _ in range(3)]
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1,3,1,1))
        self.register_buffer("pixel_std",  torch.tensor(pixel_std).view(1,3,1,1))
        self.edge_queries = edge_queries
        self.convolute_outputs = convolute_outputs
        self.enable_visualization = enable_visualization  # 添加可视化控制

        if self.convolute_outputs:
            self.output_deconv = nn.ConvTranspose2d(
                in_channels=self.sem_seg_head.predictor.num_queries + 3,
                out_channels=out_channels_1,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.layer_norm = nn.GroupNorm(
                num_groups=1, num_channels=out_channels_1
            )
            # output convolution that doesn't change dimension but just channels
            self.output_conv = nn.Conv2d(
                in_channels=out_channels_1 + 3, out_channels=out_channels_1, kernel_size=1
            )
            self.output_conv2 = nn.Conv2d(
                in_channels=out_channels_1, out_channels=1, kernel_size=1
            )
            self.activation = nn.GELU()

    def convolution_procedure(self, image, pred_gmasks):
        """
        This function is for upscaling the output of the model to the original image size.
        """
        size = image.shape[-2:]  # bs, 3, h, w
        image_res2 = F.interpolate(
            image, size=(size[0]//4, size[1]//4), mode="bilinear", align_corners=False
        )  # bs, 3, 128, 128 (stride=4)
        image_res1 = F.interpolate(
            image, size=(size[0]//2, size[1]//2), mode="bilinear", align_corners=False
        )  # bs, 3, 256, 256 (stride=2)

        mean_mask = pred_gmasks.mean(dim=1, keepdim=True)  # bs, 1, 256, 256 (stride=4)
        # bs, num_queries, 128, 128
        pred_gmasks_res1 = F.interpolate(
            mean_mask,
            size=(size[0]//2, size[1]//2),
            mode="bilinear",
            align_corners=False,
        )  # bs, 1, 256, 256

        stack_res2 = torch.cat(
            (image_res2, pred_gmasks), dim=1
        )  # bs , num_queries+3, 128, 128

        deconv_output = self.output_deconv(stack_res2)  # bs, 10, 256, 256
        deconv_output = self.layer_norm(deconv_output)  # bs, 10, 256, 256
        deconv_output = self.activation(deconv_output)  # bs, 10, 256, 256

        # concatenation with image_res2
        stack_res1 = torch.cat((image_res1, deconv_output), dim=1)  # bs, 13, 256, 256
        outputs_1_channel = self.output_conv(stack_res1)  # bs, 1, 256, 256
        outputs_1_channel = self.activation(outputs_1_channel)  # bs, 1, 256, 256
        outputs_1_channel = self.output_conv2(outputs_1_channel)  # bs, 1, 256, 256

        stacked_tensor = torch.cat((outputs_1_channel, pred_gmasks_res1), dim=1)
        averaged_results = stacked_tensor.mean(dim=1, keepdim=True)  # Take mean

        return averaged_results

    def visualize_attention(self, multi_scale_features, prompt_features, slice_start):
        """保存多尺度特征图的可视化函数（作为attention的代理）"""
        if not self.enable_visualization:
            return
            
        try:
            import matplotlib.pyplot as plt
            import gc
            
            os.makedirs("./visualizations/attention", exist_ok=True)
            
            texts = prompt_features.get('text', ['unknown'])
            
            # 可视化不同尺度的特征图作为attention热图
            for scale_idx, features in enumerate(multi_scale_features):
                batch_size = features.shape[0]
                num_prompts = len(texts)
                
                for batch_idx in range(batch_size):
                    slice_idx = slice_start + batch_idx // num_prompts
                    prompt_idx = batch_idx % num_prompts
                    
                    # 获取特征图并计算平均值作为attention map
                    feat = features[batch_idx].detach().cpu()
                    attention_map = feat.mean(dim=0).numpy()  # 在通道维度取平均
                    
                    # 保存attention热图
                    plt.figure(figsize=(8, 6))
                    plt.imshow(attention_map, cmap='hot')
                    plt.axis('off')
                    safe_text = texts[prompt_idx].replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')[:30]
                    save_name = f"attention_scale{scale_idx}_prompt{prompt_idx + 1}_{safe_text}_slice{slice_idx}.png"
                    plt.savefig(f"./visualizations/attention/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()
            
            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Attention visualization error: {e}")

    def visualize_features(self, image_embedding, slice_start):
        """保存骨干网络特征图的可视化函数"""
        if not self.enable_visualization:
            return
            
        try:
            import matplotlib.pyplot as plt
            import gc
            
            os.makedirs("./visualizations/features", exist_ok=True)
            
            if isinstance(image_embedding, dict):
                for level, feat in image_embedding.items():
                    if isinstance(feat, torch.Tensor) and len(feat.shape) == 4:
                        feat_cpu = feat[0].detach().cpu()
                        num_channels = min(8, feat_cpu.shape[0])  # 减少保存的通道数
                        
                        for i in range(num_channels):
                            plt.figure(figsize=(6, 4))
                            plt.imshow(feat_cpu[i], cmap='viridis')
                            plt.axis('off')
                            
                            save_name = f"feature_{level}_ch{i}_slice{slice_start}.png"
                            plt.savefig(f"./visualizations/features/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0)
                            plt.close()
            
            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Feature visualization error: {e}")

    def visualize_prompts(self, image_batch, outputs, prompt_features, slice_start):
        """保存文本提示结果的可视化函数"""
        if not self.enable_visualization:
            return
            
        try:
            import matplotlib.pyplot as plt
            import gc
            
            os.makedirs("./visualizations/prompts", exist_ok=True)
            
            texts = prompt_features.get('text', ['unknown'])
            pred_masks = outputs['pred_gmasks']
            
            batch_size = image_batch.shape[0]
            num_prompts = len(texts)
            
            for batch_idx in range(batch_size):
                slice_idx = slice_start + batch_idx
                
                # 获取原始图像
                img = image_batch[batch_idx].detach().cpu()
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                # 保存原始图像
                plt.figure(figsize=(6, 4))
                plt.imshow(img_np)
                plt.axis('off')
                save_name = f"original_slice{slice_idx}.png"
                plt.savefig(f"./visualizations/prompts/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # 为每个独立的prompt保存结果
                for prompt_idx, text in enumerate(texts):
                    mask_idx = batch_idx * num_prompts + prompt_idx
                    if mask_idx < pred_masks.shape[0]:
                        if len(pred_masks.shape) == 4:
                            mask = pred_masks[mask_idx, 0].detach().cpu().numpy()
                        else:
                            mask = pred_masks[mask_idx].detach().cpu().numpy()
                        
                        # 保存单独的mask
                        plt.figure(figsize=(6, 4))
                        plt.imshow(mask, cmap='hot')
                        plt.axis('off')
                        safe_text = text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')[:30]
                        save_name = f"mask_prompt{prompt_idx + 1}_{safe_text}_slice{slice_idx}.png"
                        plt.savefig(f"./visualizations/prompts/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        
                        # 保存叠加结果
                        plt.figure(figsize=(6, 4))
                        plt.imshow(img_np)
                        plt.imshow(mask, cmap='hot', alpha=0.6)
                        plt.axis('off')
                        save_name = f"result_prompt{prompt_idx + 1}_{safe_text}_slice{slice_idx}.png"
                        plt.savefig(f"./visualizations/prompts/{save_name}", dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close()
            
            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Prompt visualization error: {e}")

    def forward_train(self, inputs):
        raise NotImplementedError("Train mode is not implemented yet.")
    
        image = inputs["image"] if "image" in inputs else None
        text = inputs["text"] if "text" in inputs else None

        if image is None:
            raise ValueError("Image is required input")

        # pixel_mean = self.pixel_mean.to(image.device)
        # pixel_std = self.pixel_std.to(image.device)
        image = (image - self.pixel_mean) / self.pixel_std
        t0 = time.time()
        image_embedding = self.backbone(image)
        t1 = time.time()
        # print("backbone time: ", t1 - t0)
        
        outputs = self.sem_seg_head.forward(
            image_features=image_embedding, text=text, eval=False
        )
        t2 = time.time()
        # print("sem_seg_head time: ", t2 - t1)
        
        _, num_prompts = process_multi_prompts(text)
        num_prompts = num_prompts.to(image.device)
        outputs["num_prompts"] = num_prompts
        
        if self.convolute_outputs:
            image = image.repeat_interleave(num_prompts, dim=0)
            outputs["pred_gmasks"] = self.convolution_procedure(image, outputs["pred_gmasks"])
        else:
            outputs["pred_gmasks"] = outputs["pred_gmasks"].mean(dim=1, keepdim=True)

        results = {"predictions": outputs}
        # print("total time: ", time.time() - t0)
        return results

    def forward_eval(self, inputs, slice_batch_size):
        t0 = time.time()
        
        image = inputs["image"] if "image" in inputs else None
        text = inputs["text"] if "text" in inputs else None

        if image is None:
            raise ValueError("Image is required input")

        # check batch size is 1
        if image.shape[0] > 1:
            raise ValueError("Batch size > 1 is not supported.")
        
        # pack RGB images with neighboring slices
        if image.shape[1] == 1:
            image = image.expand(3, -1, -1, -1)  # 3, D, H, W
        else:
            image1 = torch.cat((image[:, 1:2], image[:,:-1]), dim=1)  # 1, D, H, W
            image2 = torch.cat((image[:,1:], image[:,-2:-1]), dim=1)  # 1, D, H, W
            image = torch.cat((image, image1, image2), dim=0)  # 3, D, H, W
        image = image.transpose(0, 1)  # D, 3, H, W
        
        with torch.inference_mode():
            prompt_features = self.sem_seg_head.encode_prompts(
                text=text, eval=True
            )
            
            P = int(prompt_features["num_prompts"][0])
            
            # repeat prompts features for each image in slice batch
            prompt_features["grounding_tokens"] = prompt_features["grounding_tokens"].repeat(1, slice_batch_size, 1)
            prompt_features["class_emb"] = prompt_features["class_emb"].repeat(slice_batch_size, 1)
                
            # iterate over slices in batch
            n_slices = image.shape[0]
            start = 0
            while start < n_slices:
                end = min(start + slice_batch_size, n_slices)
                image_batch = (image[start:end] - self.pixel_mean.mean()) / self.pixel_std.mean()
                
                # inference on slice batch
                image_embedding = self.backbone(image_batch)
                
                # 骨干网络特征图可视化
                self.visualize_features(image_embedding, start)
                
                if end-start < slice_batch_size:
                    # match the batch size for prompt features
                    np = (end-start) * P
                    prompt_features["grounding_tokens"] = prompt_features["grounding_tokens"][:,:np]
                    prompt_features["class_emb"] = prompt_features["class_emb"][:np]
                
                # 获取多尺度特征用于注意力可视化
                mask_features, _, multi_scale_features = (
                    self.sem_seg_head.pixel_decoder.forward_features(image_embedding)
                )
                
                # 处理特征以匹配prompt数量
                num_prompts = prompt_features["num_prompts"]
                P = int(num_prompts[0])
                if max(num_prompts) > min(num_prompts):
                    num_prompts = num_prompts.to(mask_features.device)
                    multi_scale_features = [
                        _feature.repeat_interleave(num_prompts, dim=0)
                        for _feature in multi_scale_features
                    ]
                else:
                    multi_scale_features = [
                        tile_feature(_feature, P) for _feature in multi_scale_features
                    ]
                
                # 多尺度特征作为注意力可视化
                self.visualize_attention(multi_scale_features, prompt_features, start)
                
                # forward pass
                outputs = self.sem_seg_head.forward(
                    image_features=image_embedding, prompt_features=prompt_features
                )
                
                if self.edge_queries > 0:
                    # use some of the masks for edge detection
                    outputs["edge_masks"] = outputs["pred_gmasks"][:, -self.edge_queries:].mean(dim=1, keepdim=True)
                    outputs["pred_gmasks"] = outputs["pred_gmasks"][:, :-self.edge_queries]
                else:
                    outputs["edge_masks"] = None
                
                if self.convolute_outputs:
                    image_batch = image_batch.repeat_interleave(
                        prompt_features["num_prompts"][:end-start].to(image_batch.device), dim=0
                    )
                    outputs["pred_gmasks"] = self.convolution_procedure(
                        image_batch, outputs["pred_gmasks"]
                    )
                else:
                    outputs["pred_gmasks"] = outputs["pred_gmasks"].mean(
                        dim=1, keepdim=True
                    )
                
                # 最终结果可视化
                self.visualize_prompts(image_batch, outputs, prompt_features, start)
                    
                # save outputs
                if start == 0:
                    pred_gmasks = outputs["pred_gmasks"]
                    object_existence = outputs["object_existence"]
                    if self.edge_queries > 0:
                        edge_masks = outputs["edge_masks"]
                else:
                    pred_gmasks = torch.cat((pred_gmasks, outputs["pred_gmasks"]), dim=0)
                    object_existence = torch.cat(
                        (object_existence, outputs["object_existence"]), dim=0
                    )
                    if self.edge_queries > 0:
                        edge_masks = torch.cat((edge_masks, outputs["edge_masks"]), dim=0)
                
                start += slice_batch_size
                
        # reshape to class, batch
        pred_gmasks = pred_gmasks.view(
            n_slices, P, pred_gmasks.shape[-2], pred_gmasks.shape[-1]
        )    # [D, num_prompts, H, W]
        pred_gmasks = pred_gmasks.transpose(0, 1)  # [num_prompts, D, H, W]
        
        object_existence = object_existence.view(n_slices, P).transpose(0, 1)  # [num_prompts, D]
        
        if self.edge_queries > 0:
            edge_masks = edge_masks.view(
                n_slices, P, edge_masks.shape[-2], edge_masks.shape[-1]
            )
            edge_masks = edge_masks.transpose(0, 1)  # [num_prompts, D, H, W]
            
        outputs['pred_gmasks'] = pred_gmasks
        outputs['object_existence'] = object_existence
        if self.edge_queries > 0:
            outputs['edge_masks'] = edge_masks
        outputs["inference_time"] = time.time() - t0
        
        results = {"predictions": outputs}
        
        return results
            

    def forward(self, inputs, mode="eval", slice_batch_size=2):
        if mode == "train":
            return self.forward_train(inputs)
        elif mode == "eval":
            return self.forward_eval(inputs, slice_batch_size)
        else:
            raise ValueError(f"Unknown mode {mode}. Use 'train' or 'eval'.")
        
    def load_pretrained(self, checkpoint_path):
        """Loads a pretrained checkpoint into the model."""
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
        self.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")