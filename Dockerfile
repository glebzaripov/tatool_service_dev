FROM hub.inmar.dev/library/nps-classifier:1.2

COPY ./requirements.txt /classifier/requirements.txt
WORKDIR /classifier
RUN pip3 install -r requirements.txt

COPY ./ /classifier

ENTRYPOINT ["uwsgi", "--ini", "./uwsgi.ini"]
