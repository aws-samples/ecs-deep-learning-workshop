FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get -y update

RUN apt-get -y install \
  git \
  wget \
  build-essential \
  python-dev \
  python-pip \
  libatlas-base-dev \
  libopencv-dev \
  python-opencv \
  python-numpy \
  python-matplotlib \
  python-zmq \
  libzmq-dev \
  python-setuptools \
  libcurl4-openssl-dev \
  openssh-server \
  supervisor

ENV WORKSHOPDIR /root/ecs-deep-learning-workshop

RUN mkdir ${WORKSHOPDIR}

RUN cd ${WORKSHOPDIR} \
  && git clone https://github.com/dmlc/mxnet.git --recursive \
  && git clone https://github.com/dmlc/mxnet-notebooks.git

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
  && sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '*'/g" /root/.jupyter/jupyter_notebook_config.py \
  && sed -i "s/#c.NotebookApp.password = u''/c.NotebookApp.password = u'sha1:1a7d890154f1:4f89ba21f5bcfb89c0bc2654a36900d62c262e6a'/g" \
    /root/.jupyter/jupyter_notebook_config.py

RUN mkdir -p /var/run/sshd /root/.ssh

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY id_rsa.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

WORKDIR ${WORKSHOPDIR}
EXPOSE 22 8888
CMD ["/usr/bin/supervisord"]
