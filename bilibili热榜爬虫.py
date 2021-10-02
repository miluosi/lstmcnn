def maketable(x):
    import requests
    import re
    import pandas as pd
    url= x
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.50'}
    response = requests.get(url,headers=headers)
    title = []
    trs5 = re.findall('"title":"(.*?)"', response.text)
    for p in trs5:
        title.append(p.strip().replace("&nbsp;/&nbsp;", ""))
    view=[]
    trs5 =  re.findall('"view":(.*?),',response.text)
    for p in trs5:
        view.append(p.strip().replace("&nbsp;/&nbsp;",""))
    danmu=[]
    trs5 =  re.findall('"danmaku":(.*?),',response.text)
    for p in trs5:
        danmu.append(p.strip().replace("&nbsp;/&nbsp;",""))
    reply=[]
    trs5 =  re.findall('"reply":(.*?),',response.text)
    for p in trs5:
        reply.append(p.strip().replace("&nbsp;/&nbsp;",""))
    favourite=[]
    trs5 =  re.findall('"favorite":(.*?),',response.text)
    for p in trs5:
        favourite.append(p.strip().replace("&nbsp;/&nbsp;",""))
    coin=[]
    trs5 =  re.findall('"coin":(.*?),',response.text)
    for p in trs5:
        coin.append(p.strip().replace("&nbsp;/&nbsp;",""))
    share=[]
    trs5 =  re.findall('"share":(.*?),',response.text)
    for p in trs5:
        share.append(p.strip().replace("&nbsp;/&nbsp;",""))
    now_rank=[]
    trs5 =  re.findall('"now_rank":(.*?),',response.text)
    for p in trs5:
        now_rank.append(p.strip().replace("&nbsp;/&nbsp;",""))
    his_rank=[]
    trs5 =  re.findall('"his_rank":(.*?),',response.text)
    for p in trs5:
        his_rank.append(p.strip().replace("&nbsp;/&nbsp;",""))
    like=[]
    trs5 =  re.findall('"like":(.*?),',response.text)
    for p in trs5:
        like.append(p.strip().replace("&nbsp;/&nbsp;",""))
    cid=[]
    trs5 =  re.findall('"cid":(.*?),',response.text)
    for p in trs5:
        cid.append(p.strip().replace("&nbsp;/&nbsp;",""))
    aid=[]
    trs5 =  re.findall('"aid":(.*?),',response.text)
    for p in trs5:
        aid.append(p.strip().replace("&nbsp;/&nbsp;",""))
    aid=aid[1:40:2]
    bilibili={
    '视频名':title,
    '播放量':view,
    '弹幕数量':danmu,
    '回复数量':reply,
    '收藏数':favourite,
    '投币数':coin,
    '点赞数':like,
    '分享数':share,
    '历史最高排名':his_rank,
    'cid':cid,
    'aid':aid
    }
    df =pd.DataFrame(bilibili)
    df.to_excel(r'C:\Users\86139\Desktop\bilibili.xlsx',index = False)
    return df
def getdanmu(x):
    import requests
    from lxml import etree
    import pandas as pd
    import numpy as np
    index = df[df["视频名"]== x].index.tolist()[0]
    i=df.iloc[index,:]['cid']
    i=int(i)
    url=r'https://comment.bilibili.com/%s.xml'%(i)
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.50'}
    response = requests.get(url,headers=headers)
    response.encoding = 'utf-8'
    html = etree.HTML(response.text.encode('utf-8'))
    result = html.xpath('//d/text()')
    a=np.array(result)
    a =pd.DataFrame(a)
    return a
def getpinglun(x):
    import requests
    import re
    import bs4
    from lxml import etree
    import numpy as np
    import pandas as pd
    index = df[df["视频名"]== x].index.tolist()[0]
    i=df.iloc[index,:]['aid']
    i=int(i)
    review=[]
    url='https://api.bilibili.com/x/v2/reply?pn=1&type=1&oid=%s&sort=1'%(i)
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.50'}
    response = requests.get(url,headers=headers)
    response.encoding = 'utf-8'
    review_1=re.findall('"message":"(.*?)"', response.text)
    review+=review_1
    a=np.array(review)
    a =pd.DataFrame(a)
    return a



"""
#首先运行第一个函数获得表格
#再输入视频的名字获得弹幕和评论，评论只获得了第一页的评论
x= r'https://api.bilibili.com/x/web-interface/popular?ps=20&pn=1'
maketable(x)
getdanmu("LOL让你全程发病的恐怖套路：哥谭的黑暗又回来了【有点骚东西】")
getpinglun("LOL让你全程发病的恐怖套路：哥谭的黑暗又回来了【有点骚东西】")
"""