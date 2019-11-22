import os

# 读取数据
file = open("CNKI.txt", "r")
lines = []
while True:
    article = []
    line = file.readline()  # 整行读取数据
    if not line:
        break
    else:
        lines.append(line)
file.close()
# 分段储存
articles = []
start_index = 0
end_index = 0
while True:
    # 截取正确段落至articles
    while True:
        start_line = lines[start_index]
        end_line = lines[end_index]
        if "RT" in start_line and "CNKI" in end_line:
            article = lines[start_index:end_index + 1]
            articles.append(article)
            start_index = end_index + 1
            break
        else:
            end_index += 1
    # 跳过空行
    while True:
        if start_index >= len(lines):
            break
        start_line = lines[start_index]
        if "RT" in start_line:
            end_index = start_index
            break
        else:
            start_index += 1
    if start_index >= len(lines):
        break
# 去重
new_articles = []
articles_dict = {}
for article in articles:
    a1 = None
    t1 = None
    for line in article:
        if "A1" in line:
            a1 = line
        if "T1" in line:
            t1 = line
    if (a1, t1) not in articles_dict.keys():
        articles_dict[(a1, t1)] = 1
        new_articles.append(article)
# 保存数据
file = open("new_CNKI.txt", "w")
for article in new_articles:
    for line in article:
        file.write(line)
file.close()
