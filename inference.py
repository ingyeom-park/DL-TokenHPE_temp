import os
import cv2
import torch
import csv
from torchvision import transforms
import torch.backends.cudnn as cudnn
import utils
import matplotlib
import numpy as np
import seaborn as sns
from PIL import Image
from model import TokenHPE
sns.set()

matplotlib.use('TkAgg')


if __name__ == '__main__':
	input_folder = './input'
	output_folder = './output/vis'
	csv_path      = './output/results.csv'
	
	# https://drive.google.com/file/d/1bqfJs4mvQd4jQELsj3utEEeS6SDzW30_/view
	model_path    = './weights/TokenHPEv1-ViTB-224_224-lyr3.tar'
	
	os.makedirs(input_folder, exist_ok=True)      # 추가
	os.makedirs(output_folder, exist_ok=True)     # 추가

	cudnn.enabled = True

	model = TokenHPE(num_ori_tokens=9,
				 depth=3, heads=8, embedding='sine', dim=128, inference_view=False
				 ).to("cuda")

	print('Loading data...')

	transformations = transforms.Compose([transforms.Resize(270),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	print("Loading model...")
	if model_path != "":
		saved_state_dict = torch.load(model_path, map_location='cpu')
		if 'model_state_dict' in saved_state_dict:
			model.load_state_dict(saved_state_dict['model_state_dict'])
			print("model weight loaded!")
		else:
			model.load_state_dict(saved_state_dict)
	else:
		print("model weight failed!")

	model.to("cuda")
	model.eval()

	image_extensions = ('.jpg', '.jpeg', '.png')
	image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

	csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['filename', 'pitch', 'yaw', 'roll'])

	with torch.no_grad():
		for filename in image_files:
			image_path = os.path.join(input_folder, filename)

			img = Image.open(image_path)
			img = img.convert("RGB")
			img = transformations(img)
			img = torch.unsqueeze(img, dim=0)
			img = torch.Tensor(img).to("cuda")

			R_pred, ori_9_d = model(img)

			euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
			p_pred_deg = euler[:, 0].cpu()
			y_pred_deg = euler[:, 1].cpu()
			r_pred_deg = euler[:, 2].cpu()

			pitch = round(p_pred_deg[0].item(), 2)
			yaw   = round(y_pred_deg[0].item(), 2)
			roll  = round(r_pred_deg[0].item(), 2)

			print(f"{filename} → pitch:{pitch}, yaw:{yaw}, roll:{roll}")
			csv_writer.writerow([filename, pitch, yaw, roll])

			cv2_img = cv2.imread(image_path)
			utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=100)
			save_path = os.path.join(output_folder, filename)
			cv2.imwrite(save_path, cv2_img)
			print(f"Image saved to: {save_path}")

	csv_file.close()
	print(f"\n모든 결과가 {csv_path} 에 저장됐습니다.")