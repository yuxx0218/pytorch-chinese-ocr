# -*- coding: utf-8 -*-

import uuid
import json
import cv2
import numpy as np

import web
web.config.debug  = False
render = web.template.render('templates', base='base')

from main import text_ocr
from helper.image import read_url_img,base64_to_PIL,get_now
from config import ocrPath, textPath, GPU



# load model
import torch
from model.vgg import VGG
from model.cnn import CNN

textModel = VGG(3).cuda() if GPU else VGG(3)
checkpoint = torch.load(textPath)
textModel.load_state_dict(checkpoint)
print(textModel)

ocrModel = CNN(1).cuda() if GPU else CNN(1)
checkpoint = torch.load(ocrPath)
ocrModel.load_state_dict(checkpoint)
print(ocrModel)


billList =[]
root = './test/'
timeOutTime=5


def job(uid,url,imgString,iscut,isclass,billModel,ip):
    now = get_now()
    if url is not None:
        img=read_url_img(url)
    elif imgString is not None:
        img= base64_to_PIL(imgString)
    else:
        img = None
        
    if img is not None:
        image = np.array(img)
        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        data = text_ocr(image, textModel, ocrModel)
        
        res = {'data':data,'errCode':0}
    else:
        res = {'data':[],'errCode':3}
    return res

    
class TEXT:
    """
    text detect
    """
    
    def GET(self):  
        post = {}
        now = get_now()
        ip = web.ctx.ip
        post['name'] = u''
        post['billList']      = billList
        post['postName']      = 'text'##请求地址
        post['height']        = 500
        post['width']         = 800
        post['textwidth']     = 500
        post['uuid']          = uuid.uuid1().__str__()
        post['url']           = 'text'
        return render.text(post)
    
    def POST(self): 
        post = {"errCode":0,"errMess":""}
        if 1:
            ip = web.ctx.ip
            data = web.data()
            data = json.loads(data)
            if type(data) is dict:
                uuid          = data.get('uuid')
                url           = data.get('url')
                imgString     = data.get('imgString')
                billModel     = data.get('billModel',"")
                iscut         = data.get('iscut',False)##是否多票据识别
                isclass       = data.get('isclass',False)##是否自动进行票据分类
                if 'uuid' is not None and (url is not None or imgString is not None):
                    res = job(uuid,url,imgString,iscut,isclass,billModel,ip)
                    post.update(res)
                else:
                    post["errCode"] = 1##参数缺失
                    
                
        else:
             post["errCode"]=2
        
        return json.dumps(post,ensure_ascii=False)
    


urls = ('/text','TEXT',)

  
if __name__ == "__main__":  
  
      app = web.application(urls, globals())
      app.run()
      """
      import datetime as dt
      image_path = 'test/1032838434.jpg'
      frame = cv2.imread(image_path)

      print(dt.datetime.now())
      for i in range(30):
          data = text_ocr(frame, scale, maxScale, TEXT_LINE_SCORE)
      print(data)
      print(dt.datetime.now())
      """
