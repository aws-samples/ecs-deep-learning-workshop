FROM mxnet/python

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y install git \
  python-opencv \
  build-essential \
  python-dev

RUN pip install dumb-init
RUN pip install awscli

ENV WORKSHOPDIR /root/ecs-deep-learning-workshop
RUN mkdir ${WORKSHOPDIR}

RUN cd ${WORKSHOPDIR} \
  && git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet \
  && git clone https://github.com/dmlc/mxnet-notebooks.git

COPY predict_imagenet.py /usr/local/bin/

RUN pip install jupyter

RUN jupyter-notebook --generate-config --allow-root \
  && sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '*'/g" /root/.jupyter/jupyter_notebook_config.py

ARG PASSWORD

RUN python -c "from notebook.auth import passwd;print(passwd('${PASSWORD}') if '${PASSWORD}' != '' else 'sha1:c6bd96fb0824:6654e9eabfc54d0b3d0715ddf9561bed18e09b82')" > ${WORKSHOPDIR}/password_temp

RUN sed -i "s/#c.NotebookApp.password = u''/c.NotebookApp.password = u'$(cat ${WORKSHOPDIR}/password_temp)'/g" /root/.jupyter/jupyter_notebook_config.py

RUN rm ${WORKSHOPDIR}/password_temp

WORKDIR ${WORKSHOPDIR}
EXPOSE 8888
CMD ["/usr/local/bin/dumb-init", "/usr/local/bin/jupyter-notebook", "--no-browser", "--allow-root"]