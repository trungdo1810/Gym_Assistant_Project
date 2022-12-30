import cv2
import time
import torch
import argparse
import math
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from trainer import findAngle, point_distance
from PIL import ImageFont, ImageDraw, Image

@torch.no_grad()
def run(poseweights= 'yolov7-w6-pose.pt', source='pose.mp4', device='cpu', mode='0'):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        vid_write_image = letterbox(
            cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))
        
        frame_count, total_fps = 0, 0

        # Variables
        bcount = 0
        prev_bcount = 0
        direction = 0
        time_1_rev = []
        start_time = time.time()
        
        # Load custom font
        # fontpath = "sfpro.ttf"
        fontpath = "FreeMono.ttf"
        font = ImageFont.truetype(fontpath, 20)
        font1 =  ImageFont.truetype(fontpath, 95)
        
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                # start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)

                
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Push up and counting
                if mode == '0': #mode pushup
                    # for idx in range (output.shape[0]):
                    kpts = output[0, 7:].T
                    # right arm = (5,7,9); left arm = (6,8,10)
                    # # pushup
                    # angle = findAngle(img, kpts, 5,7,9, draw=True)
                    # print(angle)
                    # percentage = np.interp(angle, (210,280), (0, 100))
                    # bar = np.interp(angle, (220,280), (int(frame_height) - 100, 100))
                    # curl
                    angle = findAngle(img, kpts, 5,7,9, draw=True)
                    print(angle)
                    percentage = np.interp(angle, (250,310), (0, 100))
                    bar = np.interp(angle, (260,310), (int(frame_height) - 100, 100))
                    # squat
                    # angle = findAngle(img, kpts, 16,14,12, draw=True)   
                    # d = point_distance(img, kpts, 15, 16, draw=True)         
                    # print(d)       
                    # percentage = np.interp(angle, (210,280), (0, 100))
                    # bar = np.interp(angle, (220,280), (int(frame_height) - 100, 100))
                    
                    color = (0, 255, 0)
                    
                    # check for push up press
                    if percentage == 100:
                        if direction == 0:
                            bcount += 0.5
                            direction = 1
                    if percentage == 0:
                        if direction == 1:
                            bcount += 0.5
                            direction = 0
                            
                    cv2.line(img, (40,100), (40, int(frame_height)-100), (255,255,255), 10)
                    cv2.line(img, (40,int(bar)), (40, int(frame_height)-100), color, 10)
                    
                    if (int(percentage)) < 10:
                        cv2.line(img, (65, int(bar)), (90, int(bar)), color, 25)
                    elif ((int(percentage)) >= 10) and (int(percentage)) < 100:
                        cv2.line(img, (65, int(bar)), (100, int(bar)), color, 25)
                    else:
                        cv2.line(img, (65, int(bar)), (110, int(bar)), color, 25)
                    
                    # break
                    
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    if bcount < 10:
                        draw.rounded_rectangle((frame_width-100, (frame_height//2)-45, frame_width - 40, (frame_height//2) + 45), fill=color, radius=20)
                        draw.text((frame_width-100, (frame_height//2)-50), f"{int(bcount)}", font=font1, fill=(0,0,0))
                    else:                        
                        draw.rounded_rectangle((frame_width-120, (frame_height//2)-45, frame_width - 10, (frame_height//2) + 45), fill=color, radius=20)
                        draw.text((frame_width-120, (frame_height//2)-50), f"{int(bcount)}", font=font1, fill=(0,0,0))
                    
                    draw.text((65, int(bar)-12), f"{int(percentage)}%", font=font, fill=(0,0,0))
                    
                    
                    # print(bcount)        
                    img = np.array(im)        
                    #=========================================================
                    
                elif mode == '2': #mode squat
                    # for idx in range (output.shape[0]):
                    kpts = output[0, 7:].T
                    # right arm = (5,7,9); left arm = (6,8,10)
                    d3 = point_distance(img, kpts, 13,14, draw=True)
                    angle = findAngle(img, kpts, 12,14,16, draw=False)                      
                    percentage = np.interp(angle, (180,280), (0, 100))
                    bar = np.interp(angle, (180,280), (int(frame_height) - 100, 100))
                    
                    color = (254, 118, 136)
                    # check for push up press
                    if percentage == 100:
                        if direction == 0:
                            bcount += 0.5
                            direction = 1
                    if percentage == 0:
                        if direction == 1:
                            bcount += 0.5
                            direction = 0
                            
                    cv2.line(img, (40,100), (40, int(frame_height)-100), (255,255,255), 10)
                    cv2.line(img, (40,int(bar)), (40, int(frame_height)-100), color, 10)
                    
                    if (int(percentage)) < 10:
                        cv2.line(img, (65, int(bar)), (90, int(bar)), color, 25)
                    elif ((int(percentage)) >= 10) and (int(percentage)) < 100:
                        cv2.line(img, (65, int(bar)), (100, int(bar)), color, 25)
                    else:
                        cv2.line(img, (65, int(bar)), (110, int(bar)), color, 25)
                    
                    # break
                    
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    if bcount < 10:
                        draw.rounded_rectangle((frame_width-100, (frame_height//2), frame_width - 50, (frame_height//2) ), fill=color, radius=20)
                    else:                        
                        draw.rounded_rectangle((frame_width-100, (frame_height//2), frame_width, (frame_height//2) ), fill=color, radius=20)
                    
                    draw.text((65, int(bar)-12), f"{int(percentage)}%", font=font, fill=(0,0,0))
                    
                    draw.text((frame_width-120, (frame_height//2)-50), f"{int(bcount)}", font=font1, fill=(0,0,0))
                    print(bcount)        
                    img = np.array(im)   
                else:
                    break
                    
                # for idx in range(output.shape[0]):
                #     plot_skeleton_kpts(img, output[idx, 7:].T, 3)

                end_time = time.time()
                if bcount - prev_bcount == 1:
                    prev_bcount = bcount                    
                    time_1_rev.append(round(end_time-start_time, 2))
                    print(time_1_rev[-1])
                    start_time = end_time
                    
                    if time_1_rev[-1] > 2.5:
                        cv2.putText(img, "Hurry up!", (frame_width-220, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
                        # draw.text((frame_width-220, 80), "Hurry up", font=font1, fill=(0,0,0))
                    elif time_1_rev[-1] < 1.0:
                        cv2.putText(img, "Slow down!", (frame_width-280, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
                        # draw.text((frame_width-220, 80), "Slow down", font=font1, fill=(0,0,0))
                    else:
                        cv2.putText(img, "Good!", (frame_width-200, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
                    # draw.text((frame_width-220, 80), "Good", font=font1, fill=(0,0,0))

                if ext.isnumeric():
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break       
                                       
                # fps = 1 / (end_time - start_time)                
                # total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()
        # frequency = bcount/time_1_rev*60
        # avg_fps = total_fps / frame_count        
        # print(f"Average FPS: {avg_fps:.3f}")
        time_1_rev.append(math.floor(bcount))
        if mode == '0':
            with open("data_pushup.txt", "w") as data:
                for i in time_1_rev:
                    data.write(str(i) + ',')
        # else:
        #     with open("data_squat.txt", "w") as data:
        #         for i in time_1_rev:
        #             data.write(str(time_1_rev) + '\n')
        
        average_time = sum(time_1_rev[1:-1])/(bcount-1)
        _, ax = plt.subplots()
        x = list(range(1,int(bcount)+1))
        y = time_1_rev[:-1]
        print(len(x))
        print(len(y))
        plt.xlabel('rev')
        plt.ylabel('time(s)')
        plt.title('Frequency')
        plt.xticks(np.arange(0, len(x)+1, 1))
        plt.yticks(np.arange(0, max(y[1:-1]), 0.1))
        
        ax.scatter(x[1:],y[1:])
        ax.axhline(y=average_time, color='red')
        plt.savefig('frequency.jpg')
        print(f"Frequecy(rev/min):{60/average_time}")
        


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='gpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--mode', type=str, default='0', help='0-pushup,weightlifting/1-squat')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)    
    main(opt)
