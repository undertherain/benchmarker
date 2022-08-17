FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends software-properties-common && \
    export TZ='Europe/Kiev' && \
    apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y --no-install-recommends locales && \
    apt-get install -y sudo  && \
    echo $TZ | tee /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV OPENCV_VERSION="3.4.2"

RUN bash install-dependencies.sh && \
    apt-get install -y docker.io
    # TODO: install docker only on CI image

RUN pip3 install -r requirements.txt && \
    bash load-data.sh && \
    rm install-dependencies.sh && \
    rm requirements.txt && \
    rm load-data.sh && \
    rm install-cv4.sh && \
    apt-get clean && \
    rm -rf ~/.cache/pip && \ 
    ls -lah ./

# TODO: pre-load dlib data