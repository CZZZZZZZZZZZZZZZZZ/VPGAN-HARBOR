from turtle import forward
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
from vlm.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from vlm.CLIP.clip import load
from options.train_options import TrainOptions
import numpy as np

opt = TrainOptions().parse()

gpu_id = opt.gpu_ids[0]

device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
#load clip
VLM_path = opt.checkpoint_dir
model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=VLM_path)
model.to(device)
for para in model.parameters():
    para.requires_grad = False


def get_clip_score(tensor,words):
	score=0
	for i in range(tensor.shape[0]):
		#image preprocess
		clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		img_resize = transforms.Resize((224,224))
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2).unsqueeze(0)
		#get probabilitis
		text = clip.tokenize(words).to(device)
		logits_per_image, logits_per_text = model(image, text)
		probs = logits_per_image.softmax(dim=-1)
		#2-word-compared probability
		# prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
		prob = probs[0][0]
		score =score + prob

	return score

clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

def get_clip_score_from_feature(tensor,text_features,num):
	score=0.
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		# image=image.reshape(1,3,224,224)
		image=clip_normalizer(image2.reshape(1,3,224,224))
		image = image.to(device)
		image_features = model.encode_image(image)
		image_nor=image_features.norm(dim=-1, keepdim=True)
		text_features = text_features.to(device)
		nor= text_features.norm(dim=-1, keepdim=True)
		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1)
		probs = similarity
		prob = probs[0][num]
		score =score + prob
	score=score/tensor.shape[0]
	return score

def get_clip_score_CE(tensor,text_features,label):
	CE_loss=0.
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		# image=image.reshape(1,3,224,224)
		image=clip_normalizer(image2.reshape(1,3,224,224))
		image = image.to(device)
		label=torch.from_numpy(np.array(label))
		
		label = label.to(device)
		image_features = model.encode_image(image)
		image_nor=image_features.norm(dim=-1, keepdim=True)
		text_features = text_features.to(device)
		nor= text_features.norm(dim=-1, keepdim=True)
		output = (100.0 * (image_features/image_nor) @ (text_features/nor).T)

		CE_loss =F.cross_entropy(output[0],label) + CE_loss
		
	return CE_loss

def get_clip_score_gan_mse(tensor1,tensor2,text_features):
	mse_loss=0.
	for i in range(tensor1.shape[0]):
		text_features = text_features.to(device)
		nor= text_features.norm(dim=-1, keepdim=True)

		image1=img_resize(tensor1[i])
		# image=image.reshape(1,3,224,224)
		image=clip_normalizer(image1.reshape(1,3,224,224))
		image = image.to(device)
		image_features1 = model.encode_image(image)
		image_nor1=image_features1.norm(dim=-1, keepdim=True)

		image2=img_resize(tensor2[i])
		# image=image.reshape(1,3,224,224)
		image2=clip_normalizer(image2.reshape(1,3,224,224))
		image2 = image2.to(device)
		image_features2 = model.encode_image(image2)
		image_nor2=image_features2.norm(dim=-1, keepdim=True)
		
		output1 = ((image_features1/image_nor1) @ (text_features/nor).T)
		output2 = ((image_features2/image_nor2) @ (text_features/nor).T)


		mse_loss = F.mse_loss(output1,output2)+ mse_loss
		
	return mse_loss

def get_clip_score_vis(tensor1,text_features):
	vis_loss=0.
	for i in range(tensor1.shape[0]):
		text_features = text_features.to(device)
		nor= text_features.norm(dim=-1, keepdim=True)

		image1=img_resize(tensor1[i])
		# image=image.reshape(1,3,224,224)
		image=clip_normalizer(image1.reshape(1,3,224,224))
		image = image.to(device)
		image_features1 = model.encode_image(image)
		image_nor1=image_features1.norm(dim=-1, keepdim=True)

		
		
		output1 = ((image_features1/image_nor1) @ (text_features/nor).T)
		


		vis_loss = vis_loss + output1
		
	return vis_loss


class L_clip_from_feature(nn.Module):
	def __init__(self):
		super(L_clip_from_feature,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, text_features,num):
		k1 = get_clip_score_from_feature(x,text_features,num)
		return k1
	


class L_clip_4class(nn.Module):
	def __init__(self):
		super(L_clip_4class,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, text_features,label):
		ce_loss = get_clip_score_CE(x,text_features,label)
		return ce_loss
	
class L_clip_gan_mse(nn.Module):
	def __init__(self):
		super(L_clip_gan_mse,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x1,x2, text_features):
		gan_mse_loss = get_clip_score_gan_mse(x1,x2,text_features)
		return gan_mse_loss

class L_clip_vis(nn.Module):
	def __init__(self):
		super(L_clip_vis,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x1,text_features):
		gan_mse_loss = get_clip_score_vis(x1,text_features)
		return gan_mse_loss



def l2_layers(pred_conv_features, input_conv_features,weight):
	weight=torch.tensor(weight).type(pred_conv_features[0].dtype)
	return weight@torch.tensor([torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
			zip(pred_conv_features, input_conv_features)],requires_grad=True)/len(weight)

