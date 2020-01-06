FROM python:3.6

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# set default screen to 1 (to match xvfb display port)
ENV DISPLAY=:1

# Python requirements for ml
COPY /requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]

EXPOSE 8888