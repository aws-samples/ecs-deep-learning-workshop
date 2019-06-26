FROM mxnet/python

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y install git \
  python-opencv \
  build-essential \
  python3-dev \
  python3-tk

RUN pip install opencv-python dumb-init awscli matplotlib

ENV WORKSHOPDIR /root/ecs-deep-learning-workshop
RUN mkdir ${WORKSHOPDIR} 

RUN cd ${WORKSHOPDIR} \
  && git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet 

COPY predict_imagenet.py /usr/local/bin/

RUN pip install jupyter

RUN jupyter-notebook --generate-config --allow-root \
  && sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '*'/g" /root/.jupyter/jupyter_notebook_config.py \
  && sed -i "s/#c.NotebookApp.allow_remote_access = False/c.NotebookApp.allow_remote_access = True/g" /root/.jupyter/jupyter_notebook_config.py

ARG PASSWORD

RUN python3 -c "from notebook.auth import passwd;print(passwd('${PASSWORD}') if '${PASSWORD}' != '' else 'sha1:c6bd96fb0824:6654e9eabfc54d0b3d0715ddf9561bed18e09b82')" > ${WORKSHOPDIR}/password_temp

RUN sed -i "s/#c.NotebookApp.password = ''/c.NotebookApp.password = '$(cat ${WORKSHOPDIR}/password_temp)'/g" /root/.jupyter/jupyter_notebook_config.py

RUN rm ${WORKSHOPDIR}/password_temp

WORKDIR ${WORKSHOPDIR}
EXPOSE 8888
CMD ["/usr/local/bin/dumb-init", "/usr/local/bin/jupyter-notebook", "--no-browser", "--allow-root"]
