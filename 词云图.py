# 导入所需库（以下库皆为anaconda3自带）
import requests  # 网页请求
import re  # 正则表达
import wordcloud  # 词云图

# 在网页中寻找自己访问网址的头部，将爬虫伪装成访问用户
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'
}

# 输入查询视频的BV号
BV = input("请输入BV号：")
# 得到该视频的网页url
BVurl = "https://m.bilibili.com/video/" + BV


##整合（定义）运行函数
def Run(BVurl):
    # 收集视频网页数据
    response1 = requests.get(BVurl, headers)

    # 视频弹幕储存另一个url请求中，需要在视频url的脚本js中进行构造
    js_str = response1.content.decode()

    # 利用正则，从获取的数据中筛选出有用部分
    data = re.findall(r'"cid":[\d]*', js_str)

    # 截取第一个数据即为所需储存弹幕url的关键信息
    data = data[0].replace('"cid":', "").replace(" ", "")

    # 构造弹幕信息的url
    url = "https://comment.bilibili.com/{}.xml".format(data)

    # 收集弹幕网页数据
    response2 = requests.get(url, headers).content.decode()

    # 利用正则获取弹幕信息
    Danmu = re.findall('<d.*?>(.*?)</d>', response2)

    # 弹幕间用空格分离，形成词云图可利用的形式
    Danmu_str = " ".join(Danmu)

    # 设置词云图参数（可更改字体、图片大小、背景颜色、词云图形状、设置黑名单等）
    w = wordcloud.WordCloud(font_path="msyh.ttc", background_color='white', width=1200, height=600)

    # 生成词云图
    w.generate(Danmu_str)

    # 将词云图保存，此处为默认保存在当前文件夹（桌面）
    w.to_file('WordCloud.png')


Run(BVurl)