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

COPY train_mnist.ipynb ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.ipynb

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

RUN sed -i "/return parser.parse_args()/i\    parser.add_argument('--log_file', type=str, default='train_output.log', help='the name of the log file that we want to log all output to')"  ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.py
RUN sed -i "/return parser.parse_args()/i\    parser.add_argument('--log_dir', type=str, default='/var/log', help='the shared log directory we will be using')" ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.py


RUN mkdir -p /var/run/sshd /root/.ssh
 
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

#COPY id_rsa /root/keys/rsa
#RUN chmod 600 /root/keys/rsa

COPY id_rsa.pub /root/.ssh/authorized_keys
#COPY config /root/.ssh/config

RUN chmod 600 /root/.ssh/authorized_keys 
#RUN eval "$(ssh-agent)" && ssh-add /root/keys/rsa

WORKDIR ${WORKSHOPDIR}
EXPOSE 22 8888 9091
CMD ["/usr/bin/supervisord"]
