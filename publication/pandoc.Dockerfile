FROM pandoc/extra

COPY packages.txt .

RUN tlmgr update --self

RUN cat packages.txt | xargs tlmgr install
