# docker build -t bigabig/wordlist:latest .
FROM python:3.10

RUN pip install streamlit spacy gensim omegaconf

WORKDIR /app

COPY ./src .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]