# specify your base image to be ubuntu:18.04
FROM ubuntu:18.04

# write down your name and email as the maintainer
LABEL maintainer="Min Zheng <businesszheng@yahoo.com>"

# install system-wide deps for python
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip python3-dev curl gnupg

# copy our application code
ADD flask-app /opt/flask-app

# set working directory to be /opt/flask-app
WORKDIR /opt/flask-app

# fetch app specific deps

# install dependencies
RUN pip3 install -r requirements.txt

# expose port
EXPOSE 5000

# start app
CMD [ "python3", "./app.py" ]