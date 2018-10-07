#FROM python:3

FROM tensorflow/tensorflow:latest-py3

ADD PIO_model ./
ADD hier_labels ./

CMD ["sh", "-c", "python ./evaluate.py participants ; python ./evaluate.py interventions ; python ./evaluate.py outcomes ; python ./evaluate1.py"]

