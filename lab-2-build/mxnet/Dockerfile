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
  openssh-server

RUN pip install dumb-init

RUN pip install awscli

ENV WORKSHOPDIR /root/ecs-deep-learning-workshop

RUN mkdir ${WORKSHOPDIR}

RUN cd ${WORKSHOPDIR} \
  && git clone https://github.com/dmlc/mxnet.git --recursive \
  && git clone https://github.com/dmlc/mxnet-notebooks.git

COPY train_mnist.ipynb ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.ipynb
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

RUN sed -i "/return parser.parse_args()/i\    parser.add_argument('--log_file', type=str, default='train_output.log', help='the name of the log file that we want to log all output to')"  ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.py \
  && sed -i "/return parser.parse_args()/i\    parser.add_argument('--log_dir', type=str, default='/var/log', help='the shared log directory we will be using')" ${WORKSHOPDIR}/mxnet/example/image-classification/train_mnist.py

WORKDIR ${WORKSHOPDIR}
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["/usr/local/bin/jupyter-notebook", "--no-browser"]
