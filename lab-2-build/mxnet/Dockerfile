FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get -y update

RUN apt-get -y install \
  git \
  wget \
  build-essential \
  python-dev \
  libatlas-base-dev \
  libopencv-dev \
  python-opencv \
  python-numpy \
  python-matplotlib \
  python-zmq \
  libzmq-dev \
  libcurl4-openssl-dev \
  openssh-server

RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

RUN pip install setuptools==33.1.1

RUN pip install dumb-init

RUN pip install awscli

ENV WORKSHOPDIR /root/ecs-deep-learning-workshop

RUN mkdir ${WORKSHOPDIR}

RUN cd ${WORKSHOPDIR} \
  && git clone https://github.com/dmlc/mxnet.git --recursive \
  && git clone https://github.com/dmlc/mxnet-notebooks.git

COPY predict_imagenet.py /usr/local/bin/

RUN cd ${WORKSHOPDIR}/mxnet \
  && cp make/config.mk . \
  && echo "USE_CUDA=0" >>config.mk \
  && echo "USE_CUDNN=0" >>config.mk \
  && echo "USE_BLAS=atlas" >>config.mk \
  && echo "USE_DIST_KVSTORE = 1" >>config.mk \
  && echo "USE_S3=1" >>config.mk \
  && make all -j$(nproc)

RUN cd ${WORKSHOPDIR}/mxnet/python \
  && python setup.py install

ENV PYTHONPATH ${WORKSHOPDIR}/mxnet/python
RUN echo "export PYTHONPATH=${WORKSHOPDIR}/mxnet/python:$PYTHONPATH" >> ~/.bashrc

RUN pip install jupyter

RUN jupyter-notebook --generate-config \
  && sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '*'/g" /root/.jupyter/jupyter_notebook_config.py

ARG PASSWORD
 
RUN python -c "from notebook.auth import passwd;print(passwd('${PASSWORD}') if '${PASSWORD}' != '' else 'sha1:c6bd96fb0824:6654e9eabfc54d0b3d0715ddf9561bed18e09b82')" > ${WORKSHOPDIR}/password_temp
 
RUN sed -i "s/#c.NotebookApp.password = u''/c.NotebookApp.password = u'$(cat ${WORKSHOPDIR}/password_temp)'/g" /root/.jupyter/jupyter_notebook_config.py

RUN rm ${WORKSHOPDIR}/password_temp

WORKDIR ${WORKSHOPDIR}
EXPOSE 8888
CMD ["/usr/local/bin/dumb-init", "/usr/local/bin/jupyter-notebook", "--no-browser"]
