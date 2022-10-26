FROM public.ecr.aws/lambda/python:3.8

COPY *.py requirements.txt ./

RUN python3 -m pip install -r requirements.txt

CMD ["backend.handler"]