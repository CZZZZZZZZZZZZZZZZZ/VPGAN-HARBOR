import os
import torch
from vlm.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from text.prompt import liver_prompts, HE_prompts, MAS_prompts,PAS_prompts,PASM_prompts



from options.train_options import TrainOptions


opt = TrainOptions().parse()

save_path = "./text/concept/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
VLM_path = opt.checkpoint_dir

conch_model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=VLM_path)
conch_model.eval()
tokenizer = get_tokenizer()
cls_templates = MAS_prompts()

feats = []
for i in range(len(cls_templates)):
            tokenized_templates = tokenize(texts=cls_templates[i], tokenizer=tokenizer)
            feats.append(conch_model.encode_text(tokenized_templates).detach())

averaged_feats = []
for feat in feats:
    # 对每个张量的行求平均
    row_averaged_feat = torch.mean(feat, dim=0, keepdim=True)
    averaged_feats.append(row_averaged_feat)
    
# 将行平均后的张量堆叠起来
stacked_feats = torch.stack(averaged_feats, dim=0)
    
# 对堆叠后的张量在第 0 维求平均
final_average = torch.mean(stacked_feats, dim=0)

print(final_average.shape)
# save feats
torch.save(final_average, os.path.join(save_path, "mas_concepts.pt"))

