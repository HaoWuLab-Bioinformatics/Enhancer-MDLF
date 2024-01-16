#基于的基础镜像
FROM python:3.8.18
# 设置code文件夹是工作目录
WORKDIR /code
#把当前文件复制到code目录下
COPY . /code
# 安装支持
RUN pip install -r requirements.txt
RUN chmod +x /code/start_project.sh

CMD ["/code/start_project.sh"]
