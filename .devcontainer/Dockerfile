FROM ubuntu:22.04
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install sudo curl wget gpg git -y
RUN apt-get install software-properties-common -y
RUN apt-get install fontconfig -y
#//For Python 3 and PIP
RUN sudo apt-get install python3 -y 
RUN sudo apt-get install python3-pip -y
#//
#//FOR Python 2 and PIP
#RUN sudo apt-get install python2.7 -y
#RUN sudo curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
#RUN sudo python2.7 get-pip.py
#//
COPY /fonts /usr/share/fonts
RUN sudo fc-cache -f -v
RUN sudo pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv $VIRTUAL_ENV --python="/usr/bin/python3"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
#COPY . .


