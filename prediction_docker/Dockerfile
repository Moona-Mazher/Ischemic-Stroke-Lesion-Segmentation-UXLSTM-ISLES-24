FROM --platform=linux/amd64 pytorch/pytorch
#FROM --platform=linux/amd64 pytorch/pytorch:latest
#FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cpu
#FROM ghcr.io/pytorch/pytorch-nightly:2.3.0.dev20240305-cuda12.1-cudnn8-runtime
#FROM --platform=linux/amd64 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app
#COPY --from=build-stage /opt/app /opt/app
# Add the directory containing the scripts to PATH
#ENV PATH="/home/user/.local/bin:$PATH"
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user nnunetv2 /opt/app/nnunetv2

#RUN mkdir -p /opt/app /input /output 
# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt /opt/app/
#RUN pip install --no-cache-dir -r /opt/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /main/.cache/pip
#RUN python -m pip install \
#    --user \
#    --no-cache-dir \
#    --no-color \
#   --requirement /opt/app/requirements.txt
COPY --chown=user:user inference.py /opt/app/
#RUN chmod a+x /workdir/my_network_infer.py
ENTRYPOINT ["python", "inference.py"]
#ENTRYPOINT [ "python", "-m", "process" ]
#ENTRYPOINT ["/opt/conda/bin/python", "/opt/app/my_network_infer.py"]
