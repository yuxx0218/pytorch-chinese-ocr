# -*- coding: utf-8 -*-

import requests
import six
from PIL import Image
import base64
import datetime as dt



def get_now():
    """
    获取当前时间
    """
    try:
        now = dt.datetime.now()
        nowString = now.strftime('%Y-%m-%d %H:%M:%S')
    except:
        nowString = '00-00-00 00:00:00'
    return nowString


def read_url_img(url):
    """
    爬取网页图片
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36'}
    try:
        req = requests.get(url,headers=headers,timeout=5)
        if req.status_code==200:
            imgString = req.content
            buf = six.BytesIO()
            buf.write(imgString)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            return img
        else:
            return None
    except:
        #traceback.print_exc()
        return None
    
    
def base64_to_PIL(string):
    try:
            
            base64_data = base64.b64decode(string.split('base64,')[-1])
            buf = six.BytesIO()
            buf.write(base64_data)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            return img
    except:
        return None
    

