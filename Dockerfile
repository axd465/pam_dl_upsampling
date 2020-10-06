FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN python3 -m pip install -U pip
RUN rm -r /tf
RUN mkdir -p /tf && chmod -R a+rwx /tf/
RUN chmod -R 755 /tmp && chmod -R 755 /var
RUN chmod a+rwx /.local

# For Tensorflow 2.1
RUN python3 -m pip install --no-cache-dir --use-feature=2020-resolver -q opencv-python-headless \
    imageio \
    tensorflow-gpu==2.1.0 \
    scikit-image \
    scikit-learn \
    pandas \
    seaborn

# Add custom user settings
RUN cd ~
RUN rm -r /root/.jupyter
COPY /.jupyter /root/.jupyter
RUN python3 -m pip install --no-cache-dir -q install jupyterlab

WORKDIR /tf
COPY ./ ./
RUN rm -rf .jupyter
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
